from dataclasses import dataclass

@dataclass
class PackedExample:
    input_ids: list
    target_ids: list

class TokenPacker:
    def __init__(self, tokenizer, seq_len, eos_id):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id

    def encode_document(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"}) + [self.eos_id]

    def pack(self, documents):
        buffer = []
        for text in documents:
            buffer.extend(self.encode_document(text))
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield chunk[:-1], chunk[1:]
