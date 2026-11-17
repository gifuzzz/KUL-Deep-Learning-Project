#!/usr/bin/env python3
"""Run all Phase 11 evaluations across CPU cores.

Models to evaluate (200 eps/tier):
1. variant_a_current: models/phase11_variant_a/final_model.zip (Current CNN+Memory)
2. variant_b_memory_only: models/phase11_variant_b/final_model.zip (Memory Only)
3. variant_c_memory_crop: models/phase11_variant_c/final_model.zip (Memory + Local Crop)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

MODELS = [
    ("variant_a_current", "models/phase11_variant_a/final_model.zip"),
    ("variant_b_memory_only", "models/phase11_variant_b/final_model.zip"),
    ("variant_c_memory_crop", "models/phase11_variant_c/final_model.zip"),
]


def main():
    log_dir = ROOT / "logs" / "phase11_evals"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "models" / "phase11_evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LAUNCHING ALL 3 PHASE 11 EVALUATIONS IN PARALLEL (200 EPS/TIER)")
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
            "--output-dir", "models/phase11_evaluations",
            "--device", "cpu",
        ]

        print(f"Starting {name} -> log: {log_path.relative_to(ROOT)}")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((name, p, log_path))

    print(f"\nAll {len(processes)} evaluation processes launched. Waiting for completion...")
    start_time = time.time()

    completed = set()
    while len(completed) < len(processes):
        time.sleep(5)
        for name, p, log_path in processes:
            if name not in completed:
                ret = p.poll()
                if ret is not None:
                    completed.add(name)
                    elapsed = time.time() - start_time
                    if ret == 0:
                        print(f"[{elapsed:6.1f}s] Process '{name}' FINISHED successfully.")
                    else:
                        print(f"[{elapsed:6.1f}s] Process '{name}' FAILED with exit code {ret}! Check {log_path.relative_to(ROOT)}")

    for f in log_files:
        f.close()

    total_time = time.time() - start_time
    print(f"\nAll evaluations complete in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
