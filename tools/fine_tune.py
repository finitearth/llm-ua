import os.path as osp
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from typing import Any

import mmengine
import torch
import torch.nn.functional as F
from alive_progress import alive_it
from bayesian_lora import calculate_kronecker_factors, cholesky_decompose_small_factors, stable_cholesky
from bayesian_lora.main import jacobian_mean
from mmengine.runner.utils import set_random_seed
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, CalibrationError
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.utils import ModelOutput

from llm_ua.datasets import DATASETS
from llm_ua.models import custom_variance, get_model_and_tokenizer


def parse_args():
    parser = ArgumentParser('Fine-tune model.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--work-dir', '-w', help='Working directory.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        '-o',
        nargs='+',
        action=mmengine.DictAction,
        help='Override the config entry using xxx=yyy format.')

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(42)
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)

    cfg = mmengine.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    logger = mmengine.MMLogger.get_instance(
        name='llm_ua',
        logger_name='llm_ua',
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    logger.info('Using config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)
    device = torch.device(f'cuda:{args.gpu_id}')

    model, tokenizer, factors = get_model_and_tokenizer(**cfg.model)

    data_set = DATASETS.build(cfg.data.data_set, default_args={'tokenizer': tokenizer})
    train_set = data_set.get_split(**cfg.data.train_split)
    # TODO to consider the case of generative tasks where there is no universal target_ids in dataset
    # TODO: add a shape checker for this target_ids, as it always needs to call squeeze. This step is easy to forget.
    target_ids = data_set.target_ids.squeeze(-1).to(device)
    val_set = data_set.get_split(**cfg.data.val_split)
    collate_fn = data_set.get_collate_fn(is_s2s=cfg.data.is_s2s)
    train_loader_cfg = cfg.data.data_loader
    train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_loader_cfg)
    val_loader_cfg = deepcopy(train_loader_cfg)
    val_loader_cfg.update({'shuffle': False})
    val_loader = DataLoader(val_set, collate_fn=collate_fn, **val_loader_cfg)

    # Fine-tune model
    if cfg.model.peft_path is None:
        assert cfg.model.get('peft_path', None) is None
        # add prior / regularization for MAP objective:
        optimizer = AdamW(model.parameters(), **cfg.optimizer)
        logger.info('Training MAP parameters')
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=50, num_training_steps=cfg.num_epochs * len(train_loader))

        for epoch in range(cfg.num_epochs):

            for batch_index, batch in alive_it(enumerate(train_loader), total=len(train_loader), enrich_print=False):
                optimizer.zero_grad()
                prompts, classes, _ = batch
                inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)
                logits = model(**inputs).logits[:, -1, target_ids]
                loss = F.cross_entropy(logits, classes.to(device))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if (batch_index + 1) % cfg.log_steps_interval == 0:
                    log_str = f'Epoch [{epoch + 1}/{cfg.num_epochs}] Batch [{batch_index + 1}/{len(train_loader)}]: '
                    log_str += f'lr: {optimizer.param_groups[0]["lr"]:.2e}, loss {loss.item():.5f}'
                    logger.info(log_str)

            # validation
            num_samples, num_correct = 0, 0
            with torch.no_grad(), torch.inference_mode():
                for batch in alive_it(val_loader, total=len(val_loader), enrich_print=False):
                    prompts, classes, _ = batch
                    inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)
                    logits = model(**inputs).logits[:, -1, target_ids]
                    probs = logits.softmax(-1)
                    num_correct += (probs.argmax(-1) == classes.to(device)).sum()
                    num_samples += logits.shape[0]

            acc = (num_correct / (num_samples + 1e-8)).item()
            logger.info(f'[Epoch [{epoch + 1}/{cfg.num_epochs}]: accuracy: {acc:.4f}')

        map_param_path = osp.join(work_dir, 'map_model_ckpt')
        logger.info(f'Saving MAP parameters after finetuning to {map_param_path}')
        model.save_pretrained(map_param_path)

    # Evaluate the log likelihood
    # ll_path = osp.join(work_dir, 'll.pth')
    # if not osp.exists(ll_path):
    #     logger.info('Evaluating the MAP log likelihood')
    #     ll = 0.0
    #     with torch.no_grad(), torch.inference_mode():
    #         for batch in alive_it(val_loader, total=len(val_loader), enrich_print=False):
    #             prompts, gt_classes, _ = batch
    #             inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)
    #             logits = model(**inputs).logits[:, -1, data_set.target_ids.squeeze(-1)]
    #             probs = logits.softmax(-1)
    #             ll += probs.gather(1, gt_classes[:, None].to(device)).sum()
    #     torch.save(ll, ll_path)
    # else:
    #     logger.info(f'Loading LL from {ll_path}')
    #     ll = torch.load(ll_path, map_location=device)

    # Calculate the (low-rank) Kronecker factors
    def forward_call(model: nn.Module, batch: Any) -> torch.Tensor:
        prompts, _, _ = batch
        inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)
        outputs = model(**inputs)
        logits = (outputs.logits[:, target_ids] if cfg.data.is_s2s else outputs.logits[:, -1, target_ids])
        return logits.softmax(-1)

    if factors is None:
        logger.info('Computing the low-rank Kronecker factors.')
        factors = calculate_kronecker_factors(
            model, forward_call, train_loader, cfg.lalora.n_kfac, cfg.lalora.lr_threshold, ['lora'], use_tqdm=True)
        # Calculate Cholesky decomposition of the smaller factors
        factors = cholesky_decompose_small_factors(factors, cfg.lalora.lr_threshold, device, torch.float32)
        kfac_path = osp.join(work_dir, 'kronecker_factors.pth')
        torch.save({'factors': factors}, kfac_path)

    # Use the marginal likelihood to optimize the prior variance
    # TODO according to our experiments, this optimization yield s2=1.009 for LLama, which leads to worse testing
    #  accuracy than the default s2=0.1. Check whether it is because the lr is too large.
    # prior_path = osp.join(work_dir, f"prior_params.pth")
    # if not osp.exists(prior_path):
    #     logger.info("Optimizing priors using marginal likelihood")
    #     s2 = torch.tensor(cfg.lalora.prior_var, requires_grad=True, device=device)
    #     optimizer = torch.optim.AdamW([s2], lr=1e-2)
    #
    #     for _ in range(200):
    #         optimizer.zero_grad()
    #         loss = model_evidence(
    #             model, ll, factors, cfg.model.peft_cfg.r, cfg.lalora.n_kfac, s2
    #         ).log()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(s2, 1.0)
    #         optimizer.step()
    #     torch.save({"s2": s2}, prior_path)
    #     logger.info(f"prior variance is: {s2.item()}")
    # else:
    #     logger.info("Loading prior parameters (optimized using marginal likelihood)")
    #     priors = torch.load(prior_path)
    #     s2 = priors["s2"]
    #     logger.info(f'Loaded prior parameters: {s2}')

    # Make linearized predictions
    logger.info('Doing linearized prediction')
    pred_mu = []
    pred_var = []
    pred_logits = []

    total_loss = 0
    metric_kwargs = {'task': 'multiclass', 'num_classes': data_set.n_labels}
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)

    def output_callback(outputs: ModelOutput) -> torch.Tensor:
        """Post process model outputs.
        """
        # Get the last token for CausalLM
        logits = outputs.logits if cfg.data.is_s2s else outputs.logits[:, -1]
        # Select the logits corresponding to our target gt_classes
        target_logits = logits[:, target_ids]
        return target_logits

    with torch.no_grad():
        logger.info('Set val_loader.batch_size = 1 to avoid CUDA OOM.')
        val_loader_cfg.update({'batch_size': 1})
        val_loader = DataLoader(val_set, collate_fn=collate_fn, **val_loader_cfg)
        # use the default s2.
        s2 = torch.tensor(cfg.lalora.prior_var, device=device)
        logger.info(f'Using s2: {s2}')

        for batch in alive_it(val_loader, total=len(val_loader), enrich_print=False):
            prompts, classes, _ = batch
            classes = classes.to(device)
            batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_cfg).to(device)

            # Predict the output logit locations
            jacobian, f_mu = jacobian_mean(model, batch_inputs, output_callback=output_callback)
            pred_mu.append(f_mu.clone().cpu())
            # Predict the output logit variances
            f_var = custom_variance(
                batch_inputs.input_ids.shape[0],
                jacobian,
                factors,
                s2,
                data_set.n_labels,
                cfg.model.peft_cfg.r,
                cfg.lalora.n_kfac,
                device,
            )
            logger.info(f'f_var shape: {f_var.shape}')
            pred_var.append(f_var.clone().cpu())
            # Sample logits from a Gaussian parametrised by f_mu, f_var
            L = stable_cholesky(f_var)
            samples = 50_000
            f_mu = f_mu.expand(samples, *f_mu.shape)
            L = L.expand(samples, *L.shape)
            eps = torch.randn_like(f_mu).unsqueeze(-1)
            logits = f_mu[..., None] + L @ eps
            logits = logits.squeeze(-1).softmax(-1).mean(0)
            pred_logits.append(logits.cpu())
            total_loss += F.cross_entropy(logits, classes).item()
            acc_metric(logits, classes)
            ece_metric(logits, classes)

    loss = total_loss / len(val_loader)
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()

    logger.info(f'NLL: {loss:.5f}, ACC: {acc:.5f}, ECE: {ece:.5f}')
    output_path = osp.join(work_dir, 'predicted_logits.pth')
    torch.save({'pred_mu': pred_mu, 'pred_var': pred_var, 'pred_logits': pred_logits}, output_path)
    logger.info('Successfully finished.')


if __name__ == '__main__':
    main()
