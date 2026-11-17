#!/usr/bin/env python3
"""Run all 6 Phase 10 evaluations concurrently across CPU cores.

Models to evaluate (200 eps/tier):
1. baseline_phase6: models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip
2. variant_a_control: models/phase10_variant_a/final_model.zip
3. variant_b_no_osc: models/phase10_variant_b/final_model.zip
4. variant_c_wait_penalty: models/phase10_variant_c/final_model.zip
5. variant_d_combined: models/phase10_variant_d/final_model.zip
6. phase9_best_nav: models/phase9_control/best_navigation_model.zip
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

MODELS = [
    ("baseline_phase6", "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip"),
    ("variant_a_control", "models/phase10_variant_a/final_model.zip"),
    ("variant_b_no_osc", "models/phase10_variant_b/final_model.zip"),
    ("variant_c_wait_penalty", "models/phase10_variant_c/final_model.zip"),
    ("variant_d_combined", "models/phase10_variant_d/final_model.zip"),
    ("phase9_best_nav", "models/phase9_control/best_navigation_model.zip"),
]


def main():
    log_dir = ROOT / "logs" / "phase10_evals"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "models" / "phase10_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LAUNCHING ALL 6 PHASE 10 EVALUATIONS IN PARALLEL (200 EPS/TIER)")
    print("=" * 80)

    processes = []
    log_files = []

    for name, path in MODELS:
        full_path = ROOT / path
        if not full_path.exists():
            print(f"ERROR: Model checkpoint not found: {full_path}")
            sys.exit(1)

        log_path = log_dir / f"{name}.log"
        log_f = open(log_path, "w")
        log_files.append(log_f)

        cmd = [
            PYTHON,
            str(ROOT / "scripts/run_phase10_evaluation_suite.py"),
            "--model", str(full_path),
            "--name", name,
            "--n-episodes", "200",
            "--device", "cpu",
            "--output-dir", "models/phase10_evaluations",
        ]

        print(f"Spawning {name} -> log: {log_path.relative_to(ROOT)}...")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
        processes.append((name, proc, log_path))

    print("\nAll 6 evaluations spawned! Monitoring progress...")
    start_time = time.time()

    # Poll until all complete
    while True:
        time.sleep(15)
        elapsed = time.time() - start_time
        all_done = True
        status_strs = []

        for name, proc, _ in processes:
            poll = proc.poll()
            if poll is None:
                all_done = False
                status_strs.append(f"{name}: RUNNING")
            else:
                status_strs.append(f"{name}: DONE (code {poll})")

        print(f"[{elapsed:6.1f}s] {' | '.join(status_strs)}")

        if all_done:
            break

    for f in log_files:
        f.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL 6 EVALUATIONS COMPLETED IN {total_time:.1f}s ({total_time / 60:.2f} min)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
