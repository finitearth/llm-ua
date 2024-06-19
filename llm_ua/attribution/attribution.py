import os.path as osp
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import mmengine
import numpy as np
import pandas as pd
import torch
from alive_progress import alive_it
from numpy import ndarray
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from transformers.utils import ModelOutput

from ..models import custom_jacobian_mean, custom_variance, entropy_of_gaussian
from ..utils import Device
from .visualization import visualize


@dataclass
class RawResult:
    """Result obtained on single input sample. All arrays have batch dimension, and the batch size is 1."""
    input_tokens: ndarray  # array of str, shape: (1, seq_len)
    input_prompts: List[str]  # list of str. The length is 1.
    attributions: Dict[str, np.ndarray]
    logits: np.ndarray  # shape: (1, num_classes)
    variance: np.ndarray  # shape: (1, num_classes, num_classes)
    gt_classes: np.ndarray  # shape: (1,)
    entropy: np.ndarray  # shape: (), means that it is a scalar array.

    def to_fp16(self):
        """Convert attributions into np.float16"""
        self.attributions = {k: v.astype(np.float16) for k, v in self.attributions.items()}
        return self

    def to_dict(self, flatten: bool) -> Dict[str, Any]:
        """Convert to dict"""
        result = asdict(self)
        if flatten:
            attributions = result.pop('attributions')
            for k, v in attributions.items():
                result.update({f'attributions_{k}': v})
        return result

    @classmethod
    def from_npz_file(cls, file_path: str) -> 'RawResult':
        """Load from a npz file"""
        with np.load(file_path, allow_pickle=True) as npz:
            instance = cls(
                input_tokens=npz['input_tokens'],
                input_prompts=npz['input_prompts'],
                attributions={
                    k.removeprefix('attributions_'): npz[k]
                    for k in npz.files if k.startswith('attributions')
                },
                logits=npz['logits'],
                variance=npz['variance'],
                gt_classes=npz['gt_classes'],
                entropy=npz['entropy'])
            return instance


class UncertaintyAttributer:

    def __init__(
            self,
            cfg: mmengine.Config,
            tokenizer: PreTrainedTokenizer,
            model: nn.Module,
            kfac_factors: Dict[str, torch.Tensor],
            target_ids: list[int],
            device: Device):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model

        logger = mmengine.MMLogger.get_current_instance()
        logger.info('Freezing model parameters.')
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.kfac_factors = kfac_factors
        self.target_ids = target_ids
        self.device = device

        self.prior_var = torch.tensor(cfg.lalora.prior_var, dtype=next(model.parameters()).dtype, device=device)

        self.lora_params = {k: v for k, v in dict(model.named_parameters()).items() if 'lora' in k.lower()}

    def output_callback(self, outputs: ModelOutput) -> torch.Tensor:
        logits = outputs.logits[:, -1, self.target_ids]
        return logits

    def _on_sample_attribution_end(self, batch_idx: int, raw_result: RawResult) -> None:
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _on_final(self, **kwargs: Any) -> None:
        pass

    def do_attribution(self, data_loader: DataLoader) -> None:
        n_labels = len(self.target_ids)
        for batch_idx, batch in alive_it(enumerate(data_loader), total=len(data_loader)):
            prompts, classes, targets = batch
            raw_result = self.get_attribution_of_sample(prompts, n_labels, targets)
            self._on_sample_attribution_end(batch_idx, raw_result)
        self._on_final()

    def get_attribution_of_sample(self, prompts: List[str], n_labels: int, classes: torch.Tensor) -> RawResult:
        inputs = self.tokenizer(prompts, **self.cfg.tokenizer_run_cfg).to(self.device)

        # TODO: currently only compatible with Peft Llama, can be wrapped to a function to consider other models.
        input_ids = inputs.pop('input_ids')
        assert input_ids.shape[0] == 1, 'batch_size should be 1.'

        # this long chain (for retrieving embed layer) is due to the wrapper of PEFT and Pretrained models.
        inputs_embeds = self.model.base_model.model.model.embed_tokens(input_ids)

        inputs_embeds.requires_grad = True
        inputs.update({'inputs_embeds': inputs_embeds})

        # f_mu.shape: (1, num_classes); hidden_states are a tuple of layer **Outputs**, including embedding layer.
        jacobian, (f_mu, hidden_states) = custom_jacobian_mean(
            self.model,
            inputs,
            lora_params=self.lora_params,
            output_callback=self.output_callback,
            output_hidden_states=True)
        # f_var.shape: (1, 5, 5)
        f_var = custom_variance(
            inputs_embeds.shape[0],
            jacobian,
            self.kfac_factors,
            self.prior_var,
            n_labels,
            self.cfg.model.peft_cfg.r,
            self.cfg.lalora.n_kfac,
            self.device,
        )

        for hs in hidden_states:
            hs.retain_grad()

        entropy = entropy_of_gaussian(f_var).mean()
        entropy.backward()

        attribution_dict = OrderedDict()
        for i, hs in enumerate(hidden_states):
            attr_layer_name = 'embed_tokens' if i == 0 else f'layers.{i-1}'
            if hs.grad is not None:
                attribution_dict.update({attr_layer_name: hs.grad.to(torch.float32).clone().detach().cpu().numpy()})

        # decode the input tokens without concatenating the decoded tokens into a single string
        # decoded_input_tokens.shape: (1, seq_len)
        decoded_input_tokens = np.asarray(self.tokenizer.batch_decode(input_ids.unsqueeze(-1)[0]))[np.newaxis, :]
        # f_mu.shape: (1, num_classes); f_var.shape: (1, num_classes, num_classes); gt_classes.shape: (1,)
        f_mu = f_mu.detach().cpu().numpy()
        f_var = f_var.detach().cpu().numpy()
        gt_classes = classes.cpu().numpy()
        entropy = entropy.detach().cpu().numpy()
        raw_result = RawResult(
            input_tokens=decoded_input_tokens,
            input_prompts=prompts,
            attributions=attribution_dict,
            logits=f_mu,
            variance=f_var,
            gt_classes=gt_classes,
            entropy=entropy)

        return raw_result


class VisualizedUA(UncertaintyAttributer):

    def __init__(
            self,
            cfg: mmengine.Config,
            tokenizer: PreTrainedTokenizer,
            model: torch.nn.Module,
            kfac_factors: dict,
            target_ids: list[int],
            device: torch.device,
            work_dir: str) -> None:
        super().__init__(cfg, tokenizer, model, kfac_factors, target_ids, device)
        self.raw_attribution_dir = osp.join(work_dir, 'raw_attributions')
        mmengine.mkdir_or_exist(self.raw_attribution_dir)
        self.vis_save_path = osp.join(work_dir, 'vis_attribution')
        mmengine.mkdir_or_exist(self.vis_save_path)

    def _on_sample_attribution_end(self, batch_idx: int, raw_result: RawResult) -> None:
        # visualize using fp64, save using fp16
        visualize(
            attribution_dict=raw_result.attributions,
            input_prompt=raw_result.input_prompts[0],
            decoded_input_tokens=raw_result.input_tokens[0].tolist(),
            uncertainty=raw_result.entropy.item(),
            logits=raw_result.logits[0],
            variance=raw_result.variance[0],
            gt_class=raw_result.gt_classes[0],
            save_path=osp.join(self.vis_save_path, f'sample_{batch_idx}.html'),
            absolute=self.cfg.visualize_cfg.absolute)
        raw_result = raw_result.to_fp16().to_dict(flatten=True)

        np.savez(osp.join(self.raw_attribution_dir, f'sample_{batch_idx}.npz'), **raw_result)


class PerAnswerUA(UncertaintyAttributer):

    def __init__(
            self,
            cfg: mmengine.Config,
            tokenizer: PreTrainedTokenizer,
            model: torch.nn.Module,
            kfac_factors: dict,
            target_ids: list[int],
            device: torch.device,
            work_dir: str,
            hidden_state: int = 0) -> None:
        super().__init__(cfg, tokenizer, model, kfac_factors, target_ids, device)
        self.df = {
            'batch_idx': [],
            'ua_question': [],
            'ua_correct_answer': [],
            'ua_mean_wrong_answers': [],
            'ua_predicted_answer': [],
            'answered_correctly': [],
            'total_entropy': []
        }

        self.work_dir = work_dir
        # TODO Question: this is ambiguous. hidden_states often refer to the hidden states tensor.
        #  But it seem to mean the index of something.
        self.hidden_state = hidden_state

    def _on_sample_attribution_end(self, batch_idx: int, raw_result: RawResult) -> None:
        ua = raw_result.attributions['embed_tokens']
        correct_class = raw_result.gt_classes
        prompts = raw_result.input_prompts
        answer_idx = self.get_answer_idx(prompts[0])

        ua_question = ua[..., self.hidden_state, :answer_idx[0]].sum()

        predicted_class = raw_result.logits.argmax(axis=-1)
        predicted_idx = answer_idx[predicted_class[0].item()]
        predicted_ua = ua[..., self.hidden_state, predicted_idx].sum()

        correct_idx = answer_idx[correct_class[0].item()]
        correct_ua = ua[..., self.hidden_state, correct_idx].sum()

        mean_ua_wrong_answers = (ua[..., self.hidden_state, :].sum() - ua_question - correct_ua) / (len(answer_idx) - 1)

        self.df['idx'].append(batch_idx)
        self.df['ua_question'].append(ua_question)
        self.df['ua_correct_answer'].append(correct_ua)
        self.df['ua_mean_wrong_answers'].append(mean_ua_wrong_answers)
        self.df['ua_predicted_answer'].append(predicted_ua)
        self.df['answered_correctly'].append(correct_class[0].item() == predicted_class[0].item())
        self.df['total_entropy'].append(raw_result.entropy.item())

    def _on_final(self):
        df = pd.DataFrame(self.df)
        df.to_csv(osp.join(self.work_dir, 'attribution_results.csv'))

    @staticmethod
    def get_answer_idx(prompt: str) -> list[int]:
        """provides the index of the first token of each answer choice in the prompt -> A), B), C), D)"""
        # TODO Question: What if the prompt questions contain A, B, C,D. E.g.
        #  Question: A laptop is a type of computer. ... A), B).
        # if A) B) C) D) are not in the prompt, use 1) 2) 3) 4) instead
        if 'D)' in prompt:
            return [prompt.index('A)'), prompt.index('B)'), prompt.index('C)'), prompt.index('D)')]
        elif 'C)' in prompt:
            return [prompt.index('A)'), prompt.index('B)'), prompt.index('C)')]
        if '4)' in prompt:
            return [prompt.index('1)'), prompt.index('2)'), prompt.index('3)'), prompt.index('4)')]
        else:
            return [prompt.index('1)'), prompt.index('2)'), prompt.index('3)')]
