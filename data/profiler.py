import argparse
import statistics
import tiktoken
from datasets import load_dataset

def profile(name, split="train", text_field="text", n=10000, tokenizer_name="gpt2"):
    tok = tiktoken.get_encoding(tokenizer_name)
    ds = load_dataset(name, split=split, streaming=True)
    lengths = []
    for row in ds:
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            lengths.append(len(tok.encode(text)))
        if len(lengths) >= n:
            break
    if not lengths:
        raise RuntimeError("No valid documents")
    lengths.sort()
    def pct(p): return lengths[min(len(lengths)-1, int(len(lengths)*p))]
    return {
        "dataset": name, "samples": len(lengths),
        "avg": statistics.mean(lengths), "p50": pct(.50),
        "p95": pct(.95), "p99": pct(.99), "max": max(lengths)
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--tokenizer", default="gpt2")
    a = p.parse_args()
    print(profile(a.dataset, n=a.n, tokenizer_name=a.tokenizer))
