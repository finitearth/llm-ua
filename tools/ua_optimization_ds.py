import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
from mmengine.runner.utils import set_random_seed
from torch.utils.data import DataLoader

from llm_ua.attribution import UncertaintyOptimizer
from llm_ua.datasets import DATASETS
from llm_ua.models import get_model_and_tokenizer


def parse_args():
    parser = ArgumentParser('Attribution per answer.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--work-dir', '-w', default='workdirs', help='Working directory.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--maximize', action='store_true', help='Maximize the entropy')
    parser.add_argument('--steps', type=int, default=10, help='Steps of optimization')
    parser.add_argument(
        '--strategy',
        type=str,
        default='bos',
        help='Which token to optimize, must be one of bos, eos, prepend or append.',
        choices=['bos', 'eos', 'prepend', 'append'])
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
    logger.info('Using config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)

    device = torch.device(f'cuda:{args.gpu_id}')
    logger.info('Using config:\n' + '=' * 60 + f'\n{cfg.pretty_text}\n' + '=' * 60)

    model, tokenizer, factors = get_model_and_tokenizer(**cfg.model, device=device)
    logger.info('Freezing model parameters.')

    dataset = DATASETS.build(cfg.data.data_set, default_args={'tokenizer': tokenizer})
    val_set = dataset.get_split(**cfg.data.val_split)
    target_ids = dataset.target_ids.squeeze(-1).to(device)

    # set bs to 1
    if cfg.data.data_loader.batch_size != 1:
        logger.info(
            f'cfg.data.data_loader.batch_size should be 1, but got {cfg.data.data_loader.batch_size}. '
            f'It is automatically set to 1.')
        cfg.data.data_loader.batch_size = 1
    data_loader = DataLoader(val_set, collate_fn=dataset.get_collate_fn(is_s2s=cfg.data.is_s2s), **cfg.data.data_loader)

    optimizer = UncertaintyOptimizer(
        cfg,
        tokenizer,
        model,
        factors,
        target_ids,
        device,
        lr=args.lr,
        steps=args.steps,
        maximize=args.maximize,
        strategy=args.strategy)

    logger.info('Starting attribution.')
    optimizer.do_attribution(data_loader)


if __name__ == '__main__':
    main()
