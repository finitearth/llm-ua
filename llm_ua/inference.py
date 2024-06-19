from typing import Callable, Dict, Optional, Tuple

import torch
from alive_progress import alive_it
from bayesian_lora import stable_cholesky
from mmengine import Config
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CalibrationError
from transformers import PreTrainedTokenizer
from transformers.utils import ModelOutput

from .attribution import FeaturePerturbation
from .models import custom_jacobian_mean, custom_variance
from .utils import Device


def get_classification_output_callback(target_ids: torch.Tensor) -> Callable[[ModelOutput], torch.Tensor]:

    assert target_ids.dim() == 1, \
        f'target_ids must be a 1D tensor with shape (num_classes, ), but got {target_ids.shape}'

    def default_output_callback(outputs: ModelOutput) -> torch.Tensor:
        """Post process model outputs.
        """
        # Get the last token for CausalLM
        target_logits = outputs.logits[:, -1, target_ids]
        return target_logits

    return default_output_callback


@torch.no_grad()
def uncertainty_aware_inference(
    model: nn.Module,
    factors: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    val_loader: DataLoader,
    target_ids: torch.Tensor,
    n_labels: int,
    cfg: Config,
    device: Device,
    feat_perturbation: Optional[FeaturePerturbation] = None,
) -> Tuple[float, float]:
    if feat_perturbation is not None:
        assert val_loader.batch_size == 1, 'batch_size should be 1 when using feature perturbation.'

    callback_fn = get_classification_output_callback(target_ids)
    s2 = torch.tensor(cfg.lalora.prior_var, device=device)
    lora_params = {k: v for k, v in dict(model.named_parameters()).items() if 'lora' in k.lower()}

    if target_ids.shape[0] != n_labels:
        raise ValueError(
            f'target_ids must have the same number of classes as the model output, '
            f'but got {target_ids.shape[0]} classes and {n_labels} logits')
    metric_kwargs = {'task': 'multiclass', 'num_classes': n_labels}
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)

    for batch_index, batch in alive_it(enumerate(val_loader), total=len(val_loader), enrich_print=False):
        prompts, classes, _ = batch
        classes = classes.to(device)
        batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)

        # register hook for feature perturbation
        if feat_perturbation is not None:
            attribution = feat_perturbation.load_attribution(batch_index=batch_index, device=device)
            handle = feat_perturbation.register_ptb_hook(attribution, model)
        else:
            handle = None

        # Predict the output logit locations
        jacobian, f_mu = custom_jacobian_mean(model, batch_inputs, lora_params=lora_params, output_callback=callback_fn)
        # Predict the output logit variances
        f_var = custom_variance(
            batch_inputs.input_ids.shape[0],
            jacobian,
            factors,
            s2,
            n_labels,
            cfg.model.peft_cfg.r,
            cfg.lalora.n_kfac,
            device,
        )
        # Sample logits from a Gaussian parametrised by f_mu, f_var
        L = stable_cholesky(f_var)
        samples = 50_000
        f_mu = f_mu.expand(samples, *f_mu.shape)
        L = L.expand(samples, *L.shape)
        eps = torch.randn_like(f_mu).unsqueeze(-1)
        logits = f_mu[..., None] + L @ eps
        logits = logits.squeeze(-1).softmax(-1).mean(0)
        acc_metric(logits, classes)
        ece_metric(logits, classes)

        if handle is not None:
            handle.remove()

    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()

    return acc, ece
