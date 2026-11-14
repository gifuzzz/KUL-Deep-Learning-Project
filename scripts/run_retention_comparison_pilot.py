#!/usr/bin/env python3
"""Run Controlled Pilot Comparison: Original Stage 2C vs Retention-Aware Stage 2C.

Compares:
1. Original Stage 2C (single near guard, 40% bridge milestone thresholds).
2. Retention-Aware Stage 2C (dual guard: near >= 85% / drop < 80%, medium >= 30% / drop < 25%,
   bridge milestone thresholds: BrA >= 20%, BrB >= 10%, BrC >= 10%).

Evaluates checkpoints at 0, 10k, 25k steps across the 6 canonical distance tiers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_curriculum import run_full_diagnosis, print_table, CURRICULUM_TIERS


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2C Retention Comparison Pilot")
    parser.add_argument("--timesteps", type=int, default=25_000, help="Timesteps per variant (default: 25,000)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per tier in diagnostic eval")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for evaluation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/controlled_pilot_stage2c",
        help="Base directory for pilot comparison outputs",
    )
    return parser.parse_args()


def run_training(
    mode: str,
    output_dir: Path,
    initial_checkpoint: Path,
    timesteps: int,
    device: str,
    seed: int,
    n_envs: int,
):
    save_dir = output_dir / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable

    cmd = [
        python_bin,
        str(PROJECT_ROOT / "training" / "train_ppo.py"),
        "--stage", "2C",
        "--stage2c-mode", mode,
        "--architecture", "baseline",
        "--total-timesteps", str(timesteps),
        "--save-dir", str(save_dir),
        "--checkpoint", str(initial_checkpoint),
        "--device", device,
        "--seed", str(seed),
        "--n-envs", str(n_envs),
        "--eval-freq", "5000",
    ]

    print(f"\n=======================================================")
    print(f"STARTING TRAINING: Stage 2C Mode = '{mode}' ({timesteps:,} steps)")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================")

    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if ret.returncode != 0:
        raise RuntimeError(f"Training failed for mode {mode} with exit code {ret.returncode}")

    print(f"Completed training '{mode}' in {elapsed:.1f}s ({timesteps / max(elapsed, 1e-3):.1f} steps/s)")
    return save_dir


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_checkpoint = PROJECT_ROOT / "models" / "curriculum_stage2" / "stage2" / "stage2_final.zip"
    if not initial_checkpoint.exists():
        raise FileNotFoundError(f"Prerequisite Stage 2 checkpoint not found: {initial_checkpoint}")

    results = {
        "metadata": {
            "timesteps": args.timesteps,
            "device": args.device,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "eval_episodes": args.eval_episodes,
            "initial_checkpoint": str(initial_checkpoint),
        },
        "step_0": {},
        "original": {},
        "retention_aware": {},
    }

    # 1. Evaluate Step 0 (Stage 2 final baseline)
    print("\n=======================================================")
    print("EVALUATING STEP 0 (Stage 2 Final Baseline)")
    print("=======================================================")
    step0_metrics = run_full_diagnosis(
        str(initial_checkpoint),
        device=args.device,
        n_episodes=args.eval_episodes,
        preset="all",
        workers=args.workers,
    )
    print_table(step0_metrics, f"Step 0 Baseline ({initial_checkpoint.name})")
    results["step_0"] = step0_metrics

    # 2. Run Original Stage 2C training (25k steps)
    orig_dir = run_training(
        mode="original",
        output_dir=output_dir,
        initial_checkpoint=initial_checkpoint,
        timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        n_envs=args.n_envs,
    )

    # 3. Run Retention-Aware Stage 2C training (25k steps)
    ret_dir = run_training(
        mode="retention_aware",
        output_dir=output_dir,
        initial_checkpoint=initial_checkpoint,
        timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        n_envs=args.n_envs,
    )

    # 4. Evaluate Checkpoints for both variants at 10k and 25k
    eval_targets = [
        ("original", "step_10k", orig_dir / "stage2c" / "checkpoint_10000.zip"),
        ("original", "step_25k", orig_dir / "stage2c" / "stage2c_final.zip"),
        ("retention_aware", "step_10k", ret_dir / "stage2c" / "checkpoint_10000.zip"),
        ("retention_aware", "step_25k", ret_dir / "stage2c" / "stage2c_final.zip"),
    ]

    for variant, step_name, cp_path in eval_targets:
        print(f"\n=======================================================")
        print(f"EVALUATING: {variant} @ {step_name} ({cp_path.name})")
        print(f"=======================================================")
        if not cp_path.exists():
            print(f"WARNING: Checkpoint {cp_path} not found!")
            continue

        metrics = run_full_diagnosis(
            str(cp_path),
            device=args.device,
            n_episodes=args.eval_episodes,
            preset="all",
            workers=args.workers,
        )
        print_table(metrics, f"{variant} @ {step_name}")
        results[variant][step_name] = metrics

    # Save complete summary JSON
    summary_path = output_dir / "pilot_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPilot comparison results saved to: {summary_path}")


if __name__ == "__main__":
    main()
