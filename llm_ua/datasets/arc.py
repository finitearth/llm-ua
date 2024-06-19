from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .base import ClassificationDataset
from .builder import DATASETS


@DATASETS.register_module()
class ARCDataset(ClassificationDataset):
    few_shot_preamble = (
        'Return the label of the correct answer for each question below.\n'
        '\n'
        'Which two body systems are directly involved in movement?\n'
        'Choices:\n'
        'A) muscular and skeletal\n'
        'B) digestive and muscular\n'
        'C) skeletal and respiratory\n'
        'E) respiratory and digestive\n'
        'Answer: A\n'
        '\n'
        '{question}\n'
        'Choices:\n'
        '{choices}\n'
        'Answer:')

    zero_shot_preamble = (
        'Return the label of the correct answer for the question below.\n'
        '\n'
        'Question: {question}\n'
        'Choices:\n'
        '{choices}\n'
        'Answer:')

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: str = 'C',
        add_space: bool = True,
        few_shot: bool = False,
    ):
        if name == 'C':
            arc_name = 'ARC-Challenge'
        elif name == 'E':
            arc_name = 'ARC-Easy'
        else:
            raise ValueError(f'Invalid ARC name: {name}')
        dset = load_dataset('ai2_arc', arc_name)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(dset, tokenizer, 5, prompt, add_space, numerical=False)

    def _format_prompts(self, batch: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for e in batch:
            choices = '\n'.join(
                [f'{label}) {text}' for text, label in zip(e['choices']['text'], e['choices']['label'])])
            prompts.append(self.preamble.format(question=e['question'], choices=choices))
        return prompts

    def clm_collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Collates a batch of data into the required format for a choice of language model training.

        Parameters:
        - batch (List[Dict[str, Any]]): A list of dictionaries, each representing a single data point in the batch.

        Returns:
        - Tuple[List[str], torch.Tensor, torch.Tensor]: A tuple containing:
            - `prompts`: A list of strings formatted as input prompts for the model.
            - `classes`: A tensor of integers where each integer indicates the index of the correct answer choice.
            - `targets`: A tensor concatenating the token IDs of the nth choice, representing the correct answer.
        """
        prompts = self._format_prompts(batch)
        classes_alpha = torch.tensor([ord(e['answerKey']) - ord('A') for e in batch])
        classes_num = []
        for e in batch:
            try:
                classes_num.append(int(e['answerKey']) - 1)
            except:
                classes_num.append(-1)
        # classes_num = t.tensor([int(e["answerKey"]) - 1 for e in batch])
        classes = torch.where(classes_alpha < 0, torch.tensor(classes_num), classes_alpha)
        targets = torch.cat([self.label2target[c.item()] for c in classes])
        return prompts, classes, targets

    def s2s_collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        prompts = self._format_prompts(batch)
        classes = torch.tensor([ord(e['answerKey']) - ord('A') for e in batch])
        targets = torch.cat([self.label2target[c.item()] for c in classes])
        # just return the target token ids
        return prompts, targets, targets
