import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
from mmengine.runner.utils import set_random_seed
from tqdm import tqdm

from llm_ua.attribution import UncertaintyAttributer
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

    sign = 1 if args.maximize else -1

    sample = 'The quick brown fox jumps over the lazy'
    target_ids = tokenizer(['dog.', 'cat.', 'horse.'], return_tensors='pt')
    target_ids = target_ids['input_ids'][:, 1]
    input_ids = tokenizer.encode(sample, return_tensors='pt').to(device)
    input_embed = model.base_model.model.model.embed_tokens(input_ids)

    uncertainty_attributer = UncertaintyAttributer(cfg, tokenizer, model, factors, target_ids, device)

    for _ in tqdm(range(args.steps)):
        raw_result = uncertainty_attributer.get_attribution_of_sample(
            inputs_embeds=input_embed, n_labels=len(target_ids), convert_to_np=False)
        grad = raw_result.attributions['embed_tokens']
        input_embed.requires_grad = False
        grad_token = grad[0, 2, :].detach()
        input_embed[0, 2, :] = input_embed[0, 2, :] + sign * grad_token * args.lr

    # find nearest neighbor in vocab
    input_embed = input_embed.detach()
    optimized_token = input_embed[0, 2, :]

    # calc distance to all tokens in the vocab and take the closest one in embeddings space
    vocab = tokenizer.get_vocab()
    vocab_keys = list(vocab.keys())
    vocab_idx = list(vocab.values())
    vocab_embed = model.base_model.model.model.embed_tokens(torch.tensor(vocab_idx).to(device))
    distances = torch.cdist(optimized_token[None, :], vocab_embed).squeeze()
    closest_token_idx = distances.argmin().item()
    closest_token = vocab_keys[closest_token_idx]

    entropy = raw_result.entropy
    logger.info(f'Original token: {sample.split()[1]}')
    logger.info(f'Clostest token idx: {closest_token_idx}')
    logger.info(f'Optimized token: {closest_token}')
    logger.info(f'new entropy: {entropy.item()}')


if __name__ == '__main__':
    main()
