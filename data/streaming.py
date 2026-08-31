from datasets import load_dataset
from .adapter import normalize_record

def stream_dataset(name, split="train", text_field="text", seed=123, shuffle_buffer=10000):
    ds = load_dataset(name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for record in ds:
        doc = normalize_record(record, text_field, name)
        if doc:
            yield doc
