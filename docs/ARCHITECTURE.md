# Architecture

HF streaming datasets
 -> document normalization
 -> weighted language mixture
 -> tokenizer
 -> EOS insertion
 -> continuous token buffer
 -> fixed 2048-token causal sequences
 -> GPT Transformer
 -> cross entropy
 -> AdamW / LR schedule
 -> validation / checkpoints / metrics
