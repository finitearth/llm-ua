from transformers import AutoTokenizer

from llm_ua.datasets import ARCDataset


def test_arc_dataset():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    data_set = ARCDataset(tokenizer)

    batch_size = 2
    data_loader = data_set.loader(batch_size=batch_size)

    batch = next(iter(data_loader))
    prompts, classes, targets = batch
    assert len(prompts) == batch_size
    assert len(classes) == batch_size
    assert len(targets) == batch_size
