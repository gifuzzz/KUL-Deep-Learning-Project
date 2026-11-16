"""Visual Debugger HTTP Server for FlameBox Agent Inspection.

Provides a local web server serving the synchronized 3-panel visual debugging interface:
- Panel 1: Ground-Truth Omniscient Simulation View
- Panel 2: Exact Agent Policy Observations (Local 9x15x15 & Remembered 9x40x52)
- Panel 3: Neural Decisions & Diagnostics (Masked probabilities, raw logits, value, GRU norm)

Supports live simulation runs, recorded trajectory browsing, deterministic replay,
and side-by-side comparison.
"""

from __future__ import annotations

import argparse
import gzip
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.recorder import (
    ACTION_NAMES,
    FEATURE_NAMES,
    MEMORY_CHANNELS,
    EpisodeRecorder,
    compute_model_hash,
    record_episode,
)
from rescue_env.wrappers import PreviousActionWrapper
from training.evaluate import load_model

ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIR = Path(__file__).parent / "viewer"
TRAJECTORIES_DIR = ROOT / "trajectories"
MODELS_DIR = ROOT / "models"


def list_checkpoints() -> list[dict[str, Any]]:
    """Scan models/ directory and return list of valid checkpoint files."""
    checkpoints = []
    if not MODELS_DIR.exists():
        return checkpoints

    for zip_path in sorted(MODELS_DIR.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel_path = str(zip_path.relative_to(ROOT))
        checkpoints.append({
            "path": rel_path,
            "filename": zip_path.name,
            "size_kb": round(zip_path.stat().st_size / 1024, 1),
            "modified": zip_path.stat().st_mtime,
            "hash": compute_model_hash(zip_path),
        })
    return checkpoints


def list_saved_trajectories() -> list[dict[str, Any]]:
    """Scan trajectories/ directory and return summary metadata of all recorded episodes."""
    trajectories = []
    if not TRAJECTORIES_DIR.exists():
        return trajectories

    # Scan both .json and .json.gz
    files = sorted(
        list(TRAJECTORIES_DIR.glob("*.json")) + list(TRAJECTORIES_DIR.glob("*.json.gz")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for fpath in files:
        try:
            if fpath.name.endswith(".gz"):
                with gzip.open(fpath, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

            meta = data.get("metadata", {})
            summary = data.get("summary", {})
            trajectories.append({
                "filename": fpath.name,
                "filepath": str(fpath.relative_to(ROOT)),
                "seed": meta.get("seed", 0),
                "tier": meta.get("tier", "Unknown"),
                "category": summary.get("category", "unclassified"),
                "category_readable": summary.get("category_readable", "Unclassified"),
                "total_steps": summary.get("total_steps", len(data.get("steps", []))),
                "total_reward": summary.get("total_reward", 0.0),
                "rescue_completed": summary.get("rescue_completed", False),
                "timeout": summary.get("timeout", False),
                "deterministic": meta.get("deterministic", True),
                "model_path": meta.get("model_path", None),
                "num_events": len(data.get("events", [])),
                "file_size_kb": round(fpath.stat().st_size / 1024, 1),
            })
        except Exception:
            continue

    return trajectories


def run_live_episode(
    model_path: str,
    seed: int = 42,
    tier: str = "Near",
    deterministic: bool = True,
    max_steps: int | None = None,
    save_trajectory: bool = True,
) -> dict[str, Any]:
    """Execute and record an episode live on the host CPU."""
    tier_lower = tier.lower().strip()
    if tier_lower in ("near", "d=2"):
        v_dist, min_dist, default_steps = 2, 1, 64
    elif tier_lower in ("medium", "d=6"):
        v_dist, min_dist, default_steps = 6, 3, 128
    elif tier_lower in ("bridge a", "bridge_a", "d=8"):
        v_dist, min_dist, default_steps = 8, 5, 160
    elif tier_lower in ("bridge b", "bridge_b", "d=10"):
        v_dist, min_dist, default_steps = 10, 7, 192
    elif tier_lower in ("bridge c", "bridge_c", "d=12"):
        v_dist, min_dist, default_steps = 12, 9, 256
    elif tier_lower in ("distant", "d=none"):
        v_dist, min_dist, default_steps = None, 1, 384
    else:
        v_dist, min_dist, default_steps = 2, 1, 64

    steps_limit = max_steps if (max_steps and max_steps > 0) else default_steps

    raw_env = RescueEnv(
        max_steps=steps_limit,
        victim_distance=v_dist,
        min_victim_distance=min_dist,
        reward_shaping=False,
        navigation_shaping=False,
    )
    env = PreviousActionWrapper(raw_env)

    target_path = ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    model, device = load_model(target_path, env=env, device="cpu")

    recorder = record_episode(
        model=model,
        env=env,
        seed=seed,
        deterministic=deterministic,
        model_path=str(target_path.relative_to(ROOT) if target_path.is_relative_to(ROOT) else target_path),
    )

    if save_trajectory:
        TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
        tier_slug = tier.replace(" ", "_").lower()
        mode_slug = "det" if deterministic else "stoch"
        out_name = f"{recorder.summary.get('category', 'run')}_{tier_slug}_s{seed}_{mode_slug}.json.gz"
        recorder.save(TRAJECTORIES_DIR / out_name)

    return recorder.to_dict()


class VisualDebuggerHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for FlameBox Visual Debugger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

    def do_GET(self):
        url_parts = urlsplit(self.path)
        path = url_parts.path
        query = parse_qs(url_parts.query)

        if path == "/api/metadata":
            self.send_json({
                "action_names": ACTION_NAMES,
                "feature_names": FEATURE_NAMES,
                "memory_channels": [
                    {"name": name, "description": desc, "color": color, "index": i}
                    for i, (name, desc, color) in enumerate(MEMORY_CHANNELS)
                ],
            })
            return

        if path == "/api/checkpoints":
            self.send_json({"checkpoints": list_checkpoints()})
            return

        if path == "/api/trajectories":
            self.send_json({"trajectories": list_saved_trajectories()})
            return

        if path == "/api/trajectory":
            filename = query.get("file", [None])[0]
            if not filename:
                self.send_error(400, "Missing 'file' parameter")
                return
            target_path = TRAJECTORIES_DIR / filename
            if not target_path.exists():
                # Try relative to ROOT
                target_path = ROOT / filename
            if not target_path.exists():
                self.send_error(404, f"Trajectory file not found: {filename}")
                return

            try:
                rec = EpisodeRecorder.load(target_path)
                self.send_json(rec.to_dict())
            except Exception as e:
                self.send_error(500, f"Error loading trajectory: {e}")
            return

        super().do_GET()

    def do_POST(self):
        url_parts = urlsplit(self.path)
        path = url_parts.path

        if path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            try:
                req_data = json.loads(post_body.decode("utf-8"))
            except Exception as e:
                self.send_error(400, f"Invalid JSON body: {e}")
                return

            model_path = req_data.get("model_path")
            if not model_path:
                self.send_error(400, "Missing 'model_path'")
                return

            seed = int(req_data.get("seed", 42))
            tier = str(req_data.get("tier", "Near"))
            deterministic = bool(req_data.get("deterministic", True))
            max_steps = req_data.get("max_steps", None)
            save_traj = bool(req_data.get("save_trajectory", True))

            try:
                result = run_live_episode(
                    model_path=model_path,
                    seed=seed,
                    tier=tier,
                    deterministic=deterministic,
                    max_steps=max_steps,
                    save_trajectory=save_traj,
                )
                self.send_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Error executing episode: {e}")
            return

        self.send_error(404, "Endpoint not found")

    def send_json(self, data: Any):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Quiet standard logs to avoid console clutter
        pass


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the local HTTP server for FlameBox Visual Debugger."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, VisualDebuggerHandler)
    print(f"\n=======================================================================")
    print(f" FlameBox Visual Agent Debugger running at: http://{host}:{port}/")
    print(f" Synchronized 3-Panel Inspection: Ground Truth, Policy Inputs, Decisions")
    print(f" Press Ctrl+C to terminate.")
    print(f"=======================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FlameBox Visual Debugger server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlameBox Visual Agent Debugger")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
