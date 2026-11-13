# Phase 4 Contract: Training Pipeline & Evaluation CLI

## 1. Curriculum Training Commands
```bash
# Full Mixed Visibility Curriculum (Stage 1 -> Gate -> Stage 2)
python -m training.train_ppo --stage all --save-dir ./models/curriculum --device mps

# Stage 1 Qualification Gate Only (250k steps, near/visible + medium/hidden)
python -m training.train_ppo --stage 1 --total-timesteps 250000 --device mps

# Stage 2 Distant Search (500k steps, multi-room) starting from Stage 1 checkpoint
python -m training.train_ppo --stage 2 --checkpoint ./models/curriculum/stage1/stage1_final.zip --total-timesteps 500000 --device mps
```

## 2. Unshaped Evaluation CLI Commands
```bash
# Deterministic qualification gate evaluation (near-rescue, unshaped, JSON output)
python -m training.evaluate --model ./models/curriculum/stage1/stage1_final.zip --episodes 100 --deterministic --victim-distance 2 --max-steps 64 --output-json ./eval_stage1_det.json

# Sampled stochastic evaluation on Stage 2 distant search (multi-room)
python -m training.evaluate --model ./models/curriculum/final_model.zip --episodes 100 --stochastic --victim-distance none --max-steps 384 --output-json ./eval_stage2_stoch.json
```
