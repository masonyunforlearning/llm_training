# Bilingual Pretraining Project

Clean-room skeleton for Korean/English causal-language-model pretraining.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\smoke_test.py
```

This repository is intentionally self-contained and does not depend on the previous project.


## Phase A-C implementation status

The repository now contains a connected pretraining path:

`configs/datasets.yaml -> Hugging Face streaming -> worker sharding -> language mixture -> tokenizer -> EOS packing -> DataLoader -> GPTModel -> AMP/gradient accumulation -> token-budget training -> checkpoint/metrics`

### Proxy smoke test

Before a 300M-token experiment, run a very small token budget:

```powershell
python training/train_proxy.py `
  --model 50M `
  --mixture ko:0.30,en:0.70 `
  --train-tokens 100000 `
  --seq-len 256 `
  --batch-size 1 `
  --grad-accum 4 `
  --lr 3e-4 `
  --num-workers 0 `
  --output-dir experiments/proxy/smoke


python -m training.train_proxy `   
  --model 50M `
  --mixture ko:0.30,en:0.70 `
  --train-tokens 100000 `
  --seq-len 256 `
  --batch-size 1 `
  --grad-accum 4 `
  --lr 3e-4 `
  --num-workers 0 `
  --output-dir experiments/proxy/smoke
  --tokenizer cl100k_base
```

This requires `configs/datasets.yaml` to contain at least one valid `ko` source. If Korean data is not configured, the command intentionally fails instead of silently training with the wrong mixture.

### Important

The current implementation does not silently fabricate a Korean dataset or validation split. Configure both explicitly after the corpus decision. If every active language has entries under `validation:`, the training CLI automatically enables evaluation.
