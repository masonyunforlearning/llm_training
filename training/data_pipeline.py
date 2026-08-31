import random
import tiktoken
import torch
from torch.utils.data import IterableDataset

class MixedTokenDataset(IterableDataset):
    def __init__(self, source_iters, weights, seq_len=2048, seed=123):
        self.source_iters=source_iters; self.weights=weights; self.seq_len=seq_len; self.seed=seed
        self.tokenizer=tiktoken.get_encoding("gpt2"); self.eos_id=self.tokenizer.eot_token
    def __iter__(self):
        info=torch.utils.data.get_worker_info()
        seed=self.seed+(info.id if info else 0)
        rng=random.Random(seed)
        iters=[iter(x) for x in self.source_iters]; buffer=[]
        while True:
            i=rng.choices(range(len(iters)),weights=self.weights,k=1)[0]
            try: doc=next(iters[i])
            except StopIteration: iters[i]=iter(self.source_iters[i]); doc=next(iters[i])
            ids=self.tokenizer.encode(doc.text,allowed_special={"<|endoftext|>"})+[self.eos_id]
            buffer.extend(ids)
            while len(buffer)>=self.seq_len+1:
                chunk=buffer[:self.seq_len+1]; buffer=buffer[self.seq_len:]
                yield torch.tensor(chunk[:-1]),torch.tensor(chunk[1:])
