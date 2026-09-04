# Kaggle 실행 가이드

## 권장 디렉터리
- Runtime cache: `/kaggle/working/token_cache` (Kaggle Output에 장기 보관하지 않음)
- Output: `/kaggle/working/output`

## 1. 환경 확인
```bash
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
!pip install -q -r requirements.txt
```

## 2. 데이터 파이프라인만 테스트
```bash
python scripts/test_data_pipeline.py --mixture ko:0.3,en:0.7 --shard-mb 64 --fill-shards 2
```

## 3. 2 GPU DDP smoke test
```bash
torchrun --standalone --nproc_per_node=2 training/train_ddp.py \
  --model small --mixture ko:0.3,en:0.7 --train-tokens 1000000 \
  --seq-len 2048 --batch-size 1 --grad-accum 8 --lr 3e-4 \
  --cache-dir /kaggle/working/token_cache --cache-gb 2 --shard-mb 64 --prefill-shards 4 \
  --output-dir /kaggle/working/output --checkpoint-interval 10 --export-interval 10
```

## 4. Resume
이전 Kaggle Output의 `output/` 폴더를 새 Notebook Input으로 연결한 뒤, 먼저 `/kaggle/input/.../output`을 `/kaggle/working/output`으로 복사하고 실행합니다.
```bash
torchrun --standalone --nproc_per_node=2 training/train_ddp.py ... --resume
```

`--portable-resume`가 기본 활성화되어 있어 runtime token cache 전체를 Output에 보존하지 않고도 resume할 수 있습니다. Rank별로 현재 unread shard suffix만 anchor로 저장합니다. 따라서 shard size를 64~128MB로 두면 Output anchor 상한도 관리하기 쉽습니다.

## Output 용량 정책
- `resume/latest.pt`: full training state, 1개만 유지
- `resume/anchors/rank*.anchor.bin`: portable resume용 최소 데이터
- `model/latest.safetensors`: optimizer 없는 모델 export
- `milestones/`: 최근 N개만 유지
- rolling token cache는 Output에 저장하지 않음
