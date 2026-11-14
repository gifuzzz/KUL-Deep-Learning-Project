#!/usr/bin/env python3
"""Ablation Pilot Experiment Runner for fire-rescue-project-2.0.

Executes Phase 5 (Initial Pilot):
Runs a controlled 50,000-step Stage 2C training pilot across the three architectures:
1. Baseline128 (1.43M params, 128-unit GRU)
2. Deep128     (2.76M params, two-stage MLP fusion, 128-unit GRU)
3. Deep256     (5.78M params, two-stage MLP fusion, 256-unit GRU, MLP heads)

Measures:
- Parameter count
- Training steps per second (throughput)
- Wall-clock training time
- Held-out 6-tier unshaped evaluation metrics (d=2, 6, 8, 10, 12, distant)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3-Architecture Pilot Comparison on Stage 2C")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["baseline", "deep128", "deep256"],
        choices=["baseline", "deep128", "deep256"],
        help="Architectures to evaluate",
    )
    parser.add_argument("--timesteps", type=int, default=50_000, help="Timesteps per variant (default: 50,000)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per tier in diagnostic eval")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/architecture_ablation",
        help="Base directory for ablation checkpoints and summaries",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    print("=" * 80)
    print("STARTING 3-ARCHITECTURE STAGE 2C PILOT ABLATION")
    print(f"Variants     : {args.architectures}")
    print(f"Step Budget  : {args.timesteps:,} steps per variant")
    print(f"Device       : {args.device}")
    print(f"Seed         : {args.seed}")
    print(f"Output Dir   : {output_dir}")
    print("=" * 80)

    summary_results = {}

    for arch in args.architectures:
        arch_dir = output_dir / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> Running Pilot Experiment: Architecture = '{arch}' ({args.timesteps:,} steps)")

        train_cmd = [
            python_bin,
            str(PROJECT_ROOT / "training" / "train_ppo.py"),
            "--stage", "2C",
            "--architecture", arch,
            "--total-timesteps", str(args.timesteps),
            "--save-dir", str(arch_dir),
            "--device", args.device,
            "--seed", str(args.seed),
            "--n-envs", str(args.n_envs),
            "--eval-freq", "5000",
        ]

        if arch == "baseline" and (PROJECT_ROOT / "models" / "curriculum_stage2" / "stage2" / "stage2_final.zip").exists():
            train_cmd.extend(["--checkpoint", str(PROJECT_ROOT / "models" / "curriculum_stage2" / "stage2" / "stage2_final.zip")])

        t_start = time.time()
        ret = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
        t_elapsed = time.time() - t_start

        if ret.returncode != 0:
            print(f"ERROR: Training failed for architecture {arch} with return code {ret.returncode}")
            continue

        steps_per_sec = float(args.timesteps) / max(t_elapsed, 1e-3)
        print(f">>> Completed {arch} in {t_elapsed:.1f}s ({steps_per_sec:.1f} steps/s)")

        # Locate final or best model
        final_model = arch_dir / "stage2c" / "stage2c_final.zip"
        if not final_model.exists():
            final_model = arch_dir / "final_model.zip"

        eval_json = arch_dir / "pilot_eval.json"
        eval_cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "diagnose_curriculum.py"),
            "--model", str(final_model),
            "--architecture", arch,
            "--episodes", str(args.eval_episodes),
            "--seed-start", "50000",
            "--tier", "all",
            "--output-json", str(eval_json),
        ]
        subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT))

        eval_data = {}
        if eval_json.exists():
            with open(eval_json, "r") as f:
                eval_data = json.load(f)

        summary_results[arch] = {
            "timesteps": args.timesteps,
            "wall_clock_seconds": round(t_elapsed, 2),
            "steps_per_second": round(steps_per_sec, 1),
            "model_path": str(final_model),
            "eval_metrics": eval_data,
        }

    # Save summary json
    summary_path = output_dir / "pilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PILOT EXPERIMENT SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Architecture':<14} | {'Params':>10} | {'Steps/s':>8} | {'Near Det%':>10} | {'d=6 Det%':>9} | {'d=8 Det%':>9} | {'d=10 Det%':>10} | {'d=12 Det%':>10} | {'Distant%':>9}"
    print(header)
    print("-" * len(header))

    params_map = {"baseline": "1,431,168", "deep128": "2,761,600", "deep256": "5,782,656"}

    for arch, data in summary_results.items():
        metrics = data.get("eval_metrics", {})
        det = metrics.get("deterministic", {})
        near = f"{det.get('Near (d=2, steps=64)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        m6 = f"{det.get('Medium (d=6, steps=128)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        b8 = f"{det.get('Bridge A (d=8, steps=160)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        b10 = f"{det.get('Bridge B (d=10, steps=192)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        b12 = f"{det.get('Bridge C (d=12, steps=256)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        dist = f"{det.get('Distant (d=None, steps=384)', {}).get('rescue_rate', 0.0)*100:.1f}%"
        sps = f"{data.get('steps_per_second', 0.0):.1f}"
        print(f"{arch:<14} | {params_map.get(arch, 'N/A'):>10} | {sps:>8} | {near:>10} | {m6:>9} | {b8:>9} | {b10:>10} | {b12:>10} | {dist:>9}")

    print("=" * 80)
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
