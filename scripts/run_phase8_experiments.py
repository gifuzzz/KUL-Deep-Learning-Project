"""Phase 8 Experiment Runner & Automated Evaluation.

Orchestrates:
1. Training Variant B (Binary Visited, 10-ch)
2. Training Variant C (Visit Recency, 10-ch, tau=40)
3. Evaluating all 5 models on Pure Exploration (200 eps), Spawn Ladder (5x200 eps), Exit Ladder (6x200 eps):
   - Phase 6 Base (9-ch base reference)
   - Variant R (Phase 7 Best 9-ch)
   - Variant A (Control: 9-ch + Corrected Reward)
   - Variant B (Binary Visited: 10-ch + Corrected Reward)
   - Variant C (Visit Recency: 10-ch + Corrected Reward)
4. Zero-ablation diagnostic on Variant B and Variant C (Channel 9 set to 0.0 at inference)
5. Aggregate summary table generation and JSON exports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_phase8_ablation import evaluate_model_full


def run_training(
    name: str,
    checkpoint: str,
    spatial_memory_mode: str,
    recency_tau: float,
    save_dir: str,
    tb_log_name: str,
    timesteps: int = 50000,
    device: str = "mps",
):
    print(f"\n=======================================================")
    print(f"Starting Training: {name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Spatial Memory Mode: {spatial_memory_mode} (tau={recency_tau})")
    print(f"Save Dir: {save_dir} | Steps: {timesteps} | Device: {device}")
    print(f"=======================================================")

    cmd = [
        sys.executable,
        "training/train_ppo.py",
        "--checkpoint", checkpoint,
        "--spatial-memory-mode", spatial_memory_mode,
        "--recency-tau", str(recency_tau),
        "--exploration-reward-mode", "normalized_continuous",
        "--save-dir", save_dir,
        "--tb-log-name", tb_log_name,
        "--stage", "3",
        "--initial-substage", "3A",
        "--distance-metric", "spawn",
        "--geometry-mode", "edge",
        "--multi-task",
        "--task-mixture", "0.32,0.28,0.20,0.20",
        "--return-mode", "route_complexity",
        "--total-timesteps", str(timesteps),
        "--stage3-timesteps", str(timesteps),
        "--eval-freq", "10000",
        "--n-eval-episodes", "20",
        "--device", device,
        "--seed", "42",
    ]

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Training for {name} failed with return code {res.returncode}")
    print(f"[SUCCESS] {name} trained in {elapsed/60.0:.2f} minutes.")


def _run_single_eval(item):
    name, path, mode, zero_ablate, out_name, episodes, device, eval_dir = item
    out_file = eval_dir / out_name
    if not Path(path).exists():
        print(f"[WARNING] Checkpoint {path} does not exist! Skipping {name}.")
        return name, None

    print(f"\n[STARTING] {name} (out: {out_name})...")
    import torch
    torch.set_num_threads(2)
    results = evaluate_model_full(
        model_path=path,
        model_name=name,
        spatial_memory_mode=mode,
        n_episodes=episodes,
        zero_ablate_ch9=zero_ablate,
        device=device,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[COMPLETED] {name} saved to {out_file}")
    return name, results


def evaluate_all_models(device: str = "cpu", episodes: int = 200):
    eval_dir = Path("models/phase8_evaluations")
    eval_dir.mkdir(parents=True, exist_ok=True)

    models_to_evaluate = [
        # 1. Base models
        ("Phase 6 Base", "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip", "none", False, "phase6_base_eval.json"),
        ("Variant R (Phase 7 Best)", "models/phase7_explore20/stage3_spawn_curriculum/best_model.zip", "none", False, "variant_r_eval.json"),
        # 2. Phase 8 ablation variants
        ("Variant A (Control)", "models/phase8_control/stage3_spawn_curriculum/best_model.zip", "none", False, "variant_a_eval.json"),
        ("Variant B (Binary Visited)", "models/phase8_binary_visited/stage3_spawn_curriculum/best_model.zip", "visited_binary", False, "variant_b_eval.json"),
        ("Variant C (Visit Recency)", "models/phase8_recency/stage3_spawn_curriculum/best_model.zip", "visit_recency", False, "variant_c_eval.json"),
        # 3. Zero-ablation diagnostic on trained memory models
        ("Variant B (Zero-Ablated Ch9)", "models/phase8_binary_visited/stage3_spawn_curriculum/best_model.zip", "visited_binary", True, "variant_b_zero_ablated_eval.json"),
        ("Variant C (Zero-Ablated Ch9)", "models/phase8_recency/stage3_spawn_curriculum/best_model.zip", "visit_recency", True, "variant_c_zero_ablated_eval.json"),
    ]

    import concurrent.futures
    items = [(name, path, mode, zero_ablate, out_name, episodes, device, eval_dir) for name, path, mode, zero_ablate, out_name in models_to_evaluate]
    all_summaries = {}

    print(f"\n[LAUNCHING PARALLEL EVALUATION] Evaluating {len(models_to_evaluate)} models across processes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(7, len(models_to_evaluate))) as executor:
        for name, results in executor.map(_run_single_eval, items):
            if results is not None:
                all_summaries[name] = results

    summary_file = eval_dir / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[ALL EVALUATIONS COMPLETE] Aggregated summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Orchestrator")
    parser.add_argument("--train-b", action="store_true", help="Train Variant B")
    parser.add_argument("--train-c", action="store_true", help="Train Variant C")
    parser.add_argument("--eval-all", action="store_true", help="Evaluate all models")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per eval tier")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    args = parser.parse_args()

    if args.train_b:
        run_training(
            name="Variant B (Binary Visited)",
            checkpoint="models/phase8_binary_visited/initial_transferred.zip",
            spatial_memory_mode="visited_binary",
            recency_tau=40.0,
            save_dir="models/phase8_binary_visited",
            tb_log_name="phase8_binary_visited",
            timesteps=50000,
            device=args.device,
        )

    if args.train_c:
        run_training(
            name="Variant C (Visit Recency)",
            checkpoint="models/phase8_recency/initial_transferred.zip",
            spatial_memory_mode="visit_recency",
            recency_tau=40.0,
            save_dir="models/phase8_recency",
            tb_log_name="phase8_recency",
            timesteps=50000,
            device=args.device,
        )

    if args.eval_all:
        evaluate_all_models(device="cpu", episodes=args.episodes)


if __name__ == "__main__":
    main()
