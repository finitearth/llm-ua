from abc import abstractmethod
from collections import OrderedDict
from typing import Callable

from torch.utils.data import Dataset


class ClassificationDataset:
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation / formatting.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        n_labels: int,
        preamble: str = '',
        add_space: bool = False,
        numerical: bool = True,
        boolean: bool = False,
    ):
        """
        Args:
            dataset: The loaded Dataset
            tokenizer: The model tokenizer
            n_labels: The number of labels / gt_classes for each question
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
            numerical: whether labels are numerical (0, 1, ...) or alphabetical (A, B, ...)
        """
        self.dataset = dataset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical

        spc = ' ' if self.add_space else ''
        """Token ids of class labels. Example [345, 673, 736]."""
        # TODO: return with enum for question type
        if numerical and boolean:
            raise ValueError('Question type cannot be both numerical and boolean')
        if boolean:
            labels = [f'{spc}True', f'{spc}False']
        elif numerical:
            labels = [f'{spc}{i}' for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A')+i)}" for i in range(self.n_labels)]
        self.target_ids = tokenizer(
            labels, return_tensors='pt',
            add_special_tokens=False).input_ids[:, -1:]  # assume these encode to single tokens
        """A mapping from label _indices_ to target token ids. This is only useful for CausalLM models.
        Example: {(0, 345), (1, 673), (2, 736)}
        """
        self.label2target = OrderedDict([(i, self.target_ids[i]) for i in range(n_labels)])
        # misnomer: should be target 2 label _index_
        self.target2label = OrderedDict([(self.target_ids[i], i) for i in range(n_labels)])

    @abstractmethod
    def s2s_collate_fn(self, batch):
        """Collate function for sequence to sequence models"""
        raise NotImplementedError

    @abstractmethod
    def clm_collate_fn(self, batch):
        """Collate function for causal language models"""
        raise NotImplementedError

    def get_split(
        self,
        split_key: str = 'train',
        subset_size: int = -1,
        subset_seed: int = 42,
    ) -> Dataset:
        if subset_size > 0:
            subset_size = (len(self.dataset[split_key]) if len(self.dataset[split_key]) < subset_size else subset_size)
            data_set = self.dataset[split_key].shuffle(seed=subset_seed).select(range(subset_size))
        else:
            data_set = self.dataset[split_key]
        return data_set

    def get_collate_fn(self, is_s2s: bool = False) -> Callable:
        if is_s2s:
            return self.s2s_collate_fn
        else:
            return self.clm_collate_fn
