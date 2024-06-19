import os.path as osp
from typing import Any

import mmengine
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from ..utils import Device
from .attribution import RawResult


class FeaturePerturbation:
    """Perturb the hidden states based on the attribution.

    Args:
        attr_data_root: The root directory of the attribution data.
        attr_layer_name: The name of the layer where the attribution is measured. This is the key stored in the
            attribution data. It might be shorter than a full layer name. For example, 'layers.0' instead of
            'model.model.layers.0'.
        model_layer_name: The name of the layer in the model to be perturbed. It should be a full layer name.
        top_k_ratio: The ratio of the top k tokens to be perturbed.
        perturb_alpha: The perturbation strength.
        verbose: Whether to log the indices of the top k tokens to be perturbed.

    """

    def __init__(
            self,
            attr_data_root: str,
            attr_layer_name: str,
            model_layer_name: str,
            top_k_ratio: float,
            perturb_alpha: float = 1.0,
            verbose: bool = False) -> None:
        self.attr_data_root = attr_data_root
        self.attr_layer_name = attr_layer_name
        self.model_layer_name = model_layer_name

        self.top_k_ratio = top_k_ratio
        self.perturb_alpha = perturb_alpha
        self.verbose = verbose

    def load_attribution(self, batch_index: int, device: Device = 'cuda:0', dtype=torch.float32) -> torch.Tensor:
        attr_path = osp.join(self.attr_data_root, f'sample_{batch_index}.npz')
        raw_result = RawResult.from_npz_file(attr_path)
        attribution = torch.tensor(raw_result.attributions[self.attr_layer_name], dtype=dtype, device=device)
        # raw attribution has shape (1, seq_len, hidden_dim) and has positive and negative values
        # take the absolute value and mean over the hidden dimension, we get the attribution of shape (1, seq_len)
        attribution = attribution.abs().mean(dim=-1)
        return attribution

    @torch.no_grad()
    def perturb(self, attribution: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Perturb the hidden states based on the attribution.

        Args:
            attribution: shape (1, seq_len). The attribution of the hidden states.
            hidden_states: shape (1, seq_len, hidden_size). The hidden states to be perturbed.

        Returns:
            ptb_hidden_states: shape (1, seq_len, hidden_size). The perturbed hidden states.
        """
        # get the top k indices of attribution
        top_k = max(1, int(hidden_states.shape[1] * self.top_k_ratio))
        top_k_indices = torch.topk(attribution, top_k, dim=-1).indices
        if self.verbose:
            logger = mmengine.MMLogger.get_current_instance()
            logger.info(
                f'Among {hidden_states.shape[1]} tokens, these indices will be perturbed: '
                f'{top_k_indices.squeeze(0)}')
        weights = torch.zeros_like(attribution)
        # scatter the perturb_alpha to the top k indices. weights: (1, seq_len, 1)
        weights.scatter_(1, top_k_indices, self.perturb_alpha)
        weights.unsqueeze_(-1)
        # mean_hidden_states: (1, 1, hidden_size)
        mean_hidden_states = hidden_states.mean(dim=1, keepdim=True)
        # ptb_hidden_states: (1, seq_len, hidden_size)
        ptb_hidden_states = hidden_states * (1.0 - weights) + mean_hidden_states * weights
        return ptb_hidden_states

    def register_ptb_hook(self, attribution: torch.Tensor, model: nn.Module) -> RemovableHandle:

        def ptb_hook(module: nn.Module, args: Any, kwargs: Any, output: Any) -> Any:
            # the attribution is measured based on the output of the model layer
            hidden_states = output[0]
            hidden_states = self.perturb(attribution, hidden_states)
            return (hidden_states, ) + output[1:]

        model_layer = model.get_submodule(self.model_layer_name)
        handle = model_layer.register_forward_hook(ptb_hook, with_kwargs=True)

        return handle
