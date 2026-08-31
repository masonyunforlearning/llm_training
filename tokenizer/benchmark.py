import argparse
import tiktoken
from datasets import load_dataset

def benchmark(dataset_name, tokenizer_names=("gpt2", "cl100k_base"), n=10000):
    ds = load_dataset(dataset_name, split="train", streaming=True)
    docs = []
    for row in ds:
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            docs.append(text)
        if len(docs) >= n: break
    chars = sum(len(x) for x in docs)
    out = []
    for name in tokenizer_names:
        tok = tiktoken.get_encoding(name)
        tokens = sum(len(tok.encode(x)) for x in docs)
        out.append({"tokenizer": name, "chars_per_token": chars/tokens, "tokens_per_char": tokens/chars})
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--n", type=int, default=10000)
    a = p.parse_args()
    for r in benchmark(a.dataset, n=a.n): print(r)
