from math import e, log, pi
from typing import Callable, Dict, Tuple, Union

import torch
from bayesian_lora.main import calc_M
from torch import Tensor, nn
from torch.func import functional_call, jacrev
from transformers import BatchEncoding
from transformers.utils import ModelOutput

from ..utils import Device


def custom_jacobian_mean(
    model: nn.Module,
    batch_inputs: BatchEncoding,
    lora_params: Dict[str, torch.Tensor],
    output_callback: Callable[[ModelOutput], Tensor] | None = None,
    output_hidden_states: bool = False,
) -> Tuple[Dict[str, Tensor], Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]]:
    """Calculates the Jacobian and logit means

    Args:
        model: the LoRA LLM from which to make predictions
        batch_inputs: the batch inputs, exactly as you would pass them into
            your model with ``model(**inputs)``.
        lora_params: The lora parameter dict.
        output_callback: a function that takes the results of
            ``model(**batch_inputs)`` and returns the logits of interest
        output_hidden_states: If true, return hidden_states of each layer.
    Returns:
        Tuple of two elements.
        - The Jacobian (a dictionary of module keys and Jacobian Tensors)
        - If output_hidden_states is True, this element is a single tensor, which represents the logit mean predictions.
            Otherwise, it is a tuple, where the first element is a tensor, representing the logit mean predictions, and
            the second element is a tuple of tensors, representing the hidden_states of all layers.
    """

    def f(model: torch.nn.Module, lora_params: Dict[str, torch.Tensor], batch_inputs: BatchEncoding):
        kwargs = {k: v for k, v in batch_inputs.items()}
        kwargs.update({'output_hidden_states': output_hidden_states})

        outputs = functional_call(model, lora_params, args=(), kwargs=kwargs)

        target_logits = output_callback(outputs)
        if output_hidden_states:
            return target_logits, (target_logits, outputs.hidden_states)
        return target_logits, target_logits

    # Calculate the Jacobian of each LoRA layer (and mean predictions)
    jacobian, f_mu = jacrev(f, argnums=1, has_aux=True)(model, lora_params, batch_inputs)
    return jacobian, f_mu


@torch.autocast('cuda', dtype=torch.float64)
def custom_variance(
    batch_size,
    jacobian: Dict[str, torch.Tensor],
    factors: Dict[str, torch.Tensor],
    s2: torch.Tensor,
    n_logits: int,
    n_lora: int,
    n_kfac: int,
    device: Device,
):
    """
    Calculates the variance matrix for performing (linearised) prediction.

    Args:
        batch_size: batch size.
        jacobian (dict): a dictionary of first derivatives for each of the
            target module's parameters
        factors: dictionary of Kronecker factors
        s2: prior variance (scalar valued tensor)
        n_logits: the number of  logits to predict (e.g. the number of gt_classes
            in your Categorical likelihood)
        n_lora: rank used in the LoRA adapters
        n_kfac: rank used for the low-rank approximation of large Kronekcer
            factors
        device: device on which to accumulate the variance matrix
    """

    # initialise a matrix to accumulate the result
    var_matrix = torch.zeros((batch_size, n_logits, n_logits), device=device)

    # Iterate over the layers; `k` is the layer name / key, `A` is the input
    # activations and `S` are the output gradients.
    for k, (A, S) in factors.items():
        # Jacobian term
        g_key = f'{k}.weight'

        # TODO: there is a squeeze operation in the original implementation. This will result in error when
        #  batch_size = 1, so we remove it. Not sure if the removal will cause further errors in some certain cases.
        G = jacobian[g_key]
        # Ensure that G is [batch, n_logits, d, n_lora] sized at all times
        if G.shape[-1] != n_lora:
            G = G.mT
        assert G.shape[-1] == n_lora

        # Flatten the last 2 dimensions; giving [batch, n_logits, d * n_lora]
        G_vec = G.flatten(-2)
        term_1 = s2 * G_vec @ G_vec.mT
        assert term_1.shape == (batch_size, n_logits, n_logits)

        M, LB = calc_M(A, S, n_lora, n_kfac, s2, return_LB=True)
        assert LB is not None
        L, B = LB
        M_size = n_kfac * n_lora
        assert M.shape == (M_size, M_size)
        M = M.to(dtype=torch.float64)

        B_expanded = B.mT[None, None, :]  # [1, 1, n_kfc, d]
        L_expanded = L[None, None, :]  # [1, 1, n_lora, n_lora]
        BGL = B_expanded @ G @ L_expanded
        BGL_vec = BGL.flatten(-2).to(dtype=torch.float64)  # [batch, n_logits, M_size]
        term_2 = s2.pow(2.0) * BGL_vec @ torch.linalg.inv(M) @ BGL_vec.mT
        assert term_2.shape == (batch_size, n_logits, n_logits)

        var_matrix += term_1 - term_2.to(var_matrix.dtype)

    return var_matrix


def entropy_of_gaussian(cov: torch.Tensor) -> torch.Tensor:
    if cov.shape[-1] != cov.shape[-2]:
        raise ValueError('Covariance matrix must be square.')

    logdet = torch.linalg.slogdet(cov).logabsdet
    if torch.isnan(logdet).any() or torch.isinf(logdet).any():
        raise RuntimeError('Covariance matrix is not positive-definite or has numerical issues.')

    k = cov.shape[-1]
    entropy = 0.5 * logdet + k / 2 * log(2 * pi * e)
    return entropy
