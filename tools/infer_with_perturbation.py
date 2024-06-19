import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
from mmengine.runner.utils import set_random_seed
from tabulate import tabulate
from torch.utils.data import DataLoader

from llm_ua.attribution import FeaturePerturbation
from llm_ua.datasets import DATASETS
from llm_ua.inference import uncertainty_aware_inference
from llm_ua.models import get_model_and_tokenizer


def parse_args():
    parser = ArgumentParser('Run inference with feature perturbation.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--work-dir', '-w', default='workdirs', help='Working directory.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument(
        '--cfg-options',
        '-o',
        nargs='+',
        action=mmengine.DictAction,
        help='Override the config entry using xxx=yyy format.')

    args = parser.parse_args()
    return args


def main():
    set_random_seed(42)
    args = parse_args()
    logger = mmengine.MMLogger.get_instance(
        name='llm_ua',
        logger_name='llm_ua',
        log_file=osp.join(args.work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'),
    )
    cfg = mmengine.Config.fromfile(args.config, format_python_code=False)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    mmengine.mkdir_or_exist(args.work_dir)
    cfg.work_dir = args.work_dir
    logger.info('Using config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)

    device = torch.device(f'cuda:{args.gpu_id}')
    model, tokenizer, factors = get_model_and_tokenizer(**cfg.model, device=device)
    logger.info('Freezing model parameters.')
    for _, param in model.named_parameters():
        param.requires_grad = False

    data_set = DATASETS.build(cfg.data.data_set, default_args={'tokenizer': tokenizer})
    target_ids = data_set.target_ids.squeeze(-1).to(device)
    val_set = data_set.get_split(**cfg.data.val_split)
    collate_fn = data_set.get_collate_fn(is_s2s=cfg.data.is_s2s)
    if cfg.data.data_loader.batch_size != 1:
        logger.info(
            f'cfg.data.data_loader.batch_size should be 1, but got {cfg.data.data_loader.batch_size}. '
            f'It is automatically set to 1.')
        cfg.data.data_loader.batch_size = 1
    if cfg.data.data_loader.get('shuffle', False):
        raise ValueError('cfg.data.data_loader.shuffle should be False.')
    val_loader = DataLoader(val_set, collate_fn=collate_fn, **cfg.data.data_loader)

    # run inference with the original model
    ori_acc, ori_ece = uncertainty_aware_inference(
        model=model, factors=factors, tokenizer=tokenizer, val_loader=val_loader, target_ids=target_ids,
        n_labels=data_set.n_labels, cfg=cfg, device=device, feat_perturbation=None)

    logger.info(f'Original model accuracy: {ori_acc:.4f}, ECE: {ori_ece:.4f}')

    # run inference with hidden_states being perturbed.
    perturbation = FeaturePerturbation(**cfg.perturbation)
    ptb_acc, ptb_ece = uncertainty_aware_inference(model=model, factors=factors, tokenizer=tokenizer,
                                                   val_loader=val_loader, target_ids=target_ids,
                                                   n_labels=data_set.n_labels,
                                                   cfg=cfg, device=device,
                                                   feat_perturbation=perturbation)
    logger.info(f'Perturbed model accuracy: {ptb_acc:.4f}, ECE: {ptb_ece:.4f}')

    tabulate_data = [
        {
            'setting': 'Original', 'acc': ori_acc, 'ece': ori_ece
        },
        {
            'setting': 'Perturbed', 'acc': ptb_acc, 'ece': ptb_ece
        },
    ]
    logger.info(f'\n{tabulate(tabulate_data, headers="keys", tablefmt="grid", floatfmt=".4f")}')


if __name__ == '__main__':
    main()
