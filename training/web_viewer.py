"""Local observation/action inspector: python -m training.web_viewer."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import BadZipFile

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rescue_env.envs.single_agent import RescueEnv


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("viewer")
ACTION_NAMES = ("Wait", "Move up", "Move right", "Move down", "Move left",
                "Pick up / drop", "Rescue outside")
FEATURE_NAMES = ("Row / height", "Column / width", "Health", "Role / 3", "Carrying victim")
NATIVE_LAYERS = (
    ("Oxygen", "Available oxygen", "#6b9faa"),
    ("Smoke", "Smoke concentration", "#85809b"),
    ("Fire", "Fire intensity", "#e27b35"),
    ("Draft", "Local air draft", "#8eaa86"),
    ("Debris", "Debris in this cell", "#b08763"),
    ("Visibility", "1 = visible; 0 = unknown", "#668fbb"),
    ("Victims", "Victim category: A = 1/3, B = 2/3, C = 1", "#bc6d98"),
    ("Other firefighters", "Other firefighters on visible cells", "#8971b8"),
    ("Passage up", "Connection from this cell to the cell above", "#388f8d"),
    ("Passage right", "Connection from this cell to the cell to the right", "#388f8d"),
    ("Passage down", "Connection from this cell to the cell below", "#388f8d"),
    ("Passage left", "Connection from this cell to the cell to the left", "#388f8d"),
    ("Smoke flammability", "Combustible fraction of smoke", "#ad795c"),
    ("Water", "Water on the ground", "#458fb8"),
    ("Material", "Material category ID, not a continuous quantity", "#a28d50"),
)


def layers_for(env):
    native = [dict(name=name, description=description, color=color, native_index=i)
              for i, (name, description, color) in enumerate(NATIVE_LAYERS)]
    selected = getattr(env, "SELECTED_CHANNELS", tuple(range(min(len(native), env.NUM_CHANNELS))))
    policy = [native[i] for i in selected]
    policy.append(dict(name="Visible exits", description="Visible interior exit cells",
                       color="#36996a", native_index=None))
    return {"policy": policy, "native": native}


class Inspector:
    def __init__(self, seed=42, max_steps=64, victim_distance=2):
        self.env = RescueEnv(max_steps=max_steps, victim_distance=victim_distance,
                             reward_shaping=True)
        self.model = None
        self.checkpoint = None
        self.reset(seed)

    def reset(self, seed, shaping=None):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Seed must be an integer between 0 and 2147483647")
        if shaping is not None:
            if not isinstance(shaping, bool):
                raise ValueError("Reward shaping must be true or false")
            self.env.reward_shaping = shaping
        self.observation, self.info = self.env.reset(seed=seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.total_reward = 0.0
        self.finished = False
        self.transition = None
        return self.state()

    def checkpoints(self):
        paths = sorted((ROOT / "models").rglob("*.zip"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(p.relative_to(ROOT)) for p in paths]

    def load(self, checkpoint):
        if checkpoint is None:
            self.model, self.checkpoint = None, None
            return self.reset(self.seed)
        if not isinstance(checkpoint, str):
            raise ValueError("Choose a checkpoint from the model list")
        path = (ROOT / checkpoint).resolve()
        if not path.is_relative_to((ROOT / "models").resolve()) or path.suffix != ".zip":
            raise ValueError("Checkpoint must be a .zip file inside this project's models directory")
        model = MaskablePPO.load(path, device="cpu")
        navigation = model.observation_space["agent"].shape == (8,)
        candidate = self.env
        if navigation != getattr(self.env, "navigation", False):
            candidate = RescueEnv(max_steps=1500 if navigation else 64,
                                  victim_distance=40 if navigation else 2,
                                  min_victim_distance=20 if navigation else 1,
                                  reward_shaping=self.env.reward_shaping)
        if (model.observation_space != candidate.observation_space
                or model.action_space != candidate.action_space):
            if candidate is not self.env:
                candidate.close()
            raise ValueError(
                f"Checkpoint inputs do not match this environment. Expected grid "
                f"{self.env.observation_space['grid'].shape} and agent "
                f"{self.env.observation_space['agent'].shape}; got "
                f"{model.observation_space['grid'].shape} and "
                f"{model.observation_space['agent'].shape}. Choose a newer checkpoint."
            )
        model.policy.set_training_mode(False)
        if candidate is not self.env:
            self.env.close()
            self.env = candidate
        self.model, self.checkpoint = model, str(path.relative_to(ROOT.resolve()))
        return self.reset(self.seed)

    def policy_output(self, masks):
        if self.model is None:
            return None
        policy = self.model.policy
        with torch.no_grad():
            tensors, _ = policy.obs_to_tensor(self.observation)
            features = policy.extract_features(tensors)
            if isinstance(features, tuple):
                actor_features, critic_features = features
                latent_pi = policy.mlp_extractor.forward_actor(actor_features)
                latent_vf = policy.mlp_extractor.forward_critic(critic_features)
            else:
                actor_features = critic_features = features
                latent_pi, latent_vf = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)[0].tolist()
            distribution = policy._get_action_dist_from_latent(latent_pi)
            unmasked = distribution.distribution.probs[0].tolist()
            distribution.apply_masking(masks)
            probabilities = distribution.distribution.probs[0].tolist()
            return {
                "logits": logits,
                "unmasked_probabilities": unmasked,
                "probabilities": probabilities,
                "log_probabilities": distribution.distribution.logits[0].tolist(),
                "value": float(policy.value_net(latent_vf)[0, 0]),
                "features": actor_features[0].tolist(),
                "critic_features": critic_features[0].tolist(),
                "suggested_action": int(np.argmax(probabilities)),
            }

    def state(self):
        e = self.env.sim
        masks = self.env.action_masks()
        native = e.observe(agent_id=0, visibility_radius=self.env.visibility_radius)
        walls = [[int(c + 1), int(r), int(c + 1), int(r + 1)]
                 for r, c in np.argwhere(e.east < .5)]
        walls += [[int(c), int(r + 1), int(c + 1), int(r + 1)]
                  for r, c in np.argwhere(e.south < .5)]
        openings = []
        for o in e.openings:
            at, start, length = o["at"], o["start"], o["length"]
            line = [start, at, start + length, at] if o["orient"] == "H" else [at, start, at, start + length]
            openings.append({"line": line, "kind": o["kind"], "open": o["open"],
                             "blocked": bool(o.get("blocked"))})
        return {
            "seed": self.seed, "step": self.env.steps, "ticks": e.ticks,
            "max_steps": self.env.max_steps, "finished": self.finished,
            "total_reward": self.total_reward, "reward_shaping": self.env.reward_shaping,
            "checkpoint": self.checkpoint,
            "trained_steps": self.model.num_timesteps if self.model else None,
            "checkpoints": self.checkpoints(), "layers": layers_for(self.env),
            "feature_names": (FEATURE_NAMES + ("Waypoint row delta", "Waypoint column delta",
                                                "Navigation subtask") if getattr(self.env, "navigation", False)
                              else FEATURE_NAMES), "action_names": ACTION_NAMES,
            "native_commands": self.env.ACTIONS,
            "observation": {key: value.tolist() for key, value in self.observation.items()},
            "native_observation": {key: value.tolist() for key, value in native.items()},
            "action_masks": masks.tolist(), "policy": self.policy_output(masks),
            "info": self.info, "transition": self.transition,
            "scene": {"h": e.h, "w": e.w, "margin": 3 * e.resolution,
                      "walls": walls, "openings": openings, "zone": e.zone.tolist(),
                      "fire": e.fire.tolist(), "smoke": e.smoke.tolist(),
                      "water": e.water_level.tolist(), "debris": e.debris.tolist(),
                      "agents": e.agents, "victims": e.victims},
        }

    def step(self, action=None, mode="greedy", allow_invalid=False):
        if self.finished:
            raise ValueError("Episode finished. Reset before taking another action.")
        if mode not in ("greedy", "sampled") or not isinstance(allow_invalid, bool):
            raise ValueError("Invalid action selection settings")
        before = self.state()
        masks = before["action_masks"]
        output = before["policy"]
        source = "manual"
        if action is None:
            if output is None:
                raise ValueError("Load a checkpoint to let the policy choose an action")
            source = mode
            p = np.asarray(output["probabilities"], dtype=float)
            action = int(p.argmax()) if mode == "greedy" else int(self.rng.choice(len(p), p=p / p.sum()))
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(ACTION_NAMES):
            raise ValueError("Action must be an integer from 0 to 6")
        if not masks[action] and not allow_invalid:
            raise ValueError("That action is masked. Enable 'Test masked actions' to inspect its result.")

        native_result = None
        original_batch = self.env.sim.action_batch

        def record_batch(*args, **kwargs):
            nonlocal native_result
            native_result = original_batch(*args, **kwargs)
            return native_result

        # Capture this inspector's simulator output without changing the training environment.
        self.env.sim.action_batch = record_batch
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        finally:
            self.env.sim.action_batch = original_batch
        self.observation, self.info = obs, info
        self.finished = bool(terminated or truncated)
        self.total_reward += reward
        self.transition = {
            "step": self.env.steps, "source": source, "action": action,
            "action_name": ACTION_NAMES[action], "native_command": self.env.ACTIONS[action],
            "was_valid": masks[action], "input": before["observation"],
            "native_input": before["native_observation"],
            "action_masks": masks, "policy_output": output,
            "reward": reward, "native_reward": native_result["reward"],
            "shaping_adjustment": reward - native_result["reward"],
            "terminated": bool(terminated), "truncated": bool(truncated),
            "info": info, "native_output": native_result,
            "agent_before": before["scene"]["agents"][0],
        }
        return self.state()

    def close(self):
        self.env.close()


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != host or self.headers.get("Origin", f"http://{host}") != f"http://{host}":
                self.reply(403, {"error": "Use this viewer's http://127.0.0.1 address"})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path == "/api/state":
                self.reply(200, viewer.state())
            elif path in files:
                name, content_type = files[path]
                self.reply(200, (ASSETS / name).read_bytes(), content_type)
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local_request():
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/reset":
                    result = viewer.reset(data.get("seed", viewer.seed), data.get("shaping"))
                elif self.path == "/api/step":
                    result = viewer.step(data.get("action"), data.get("mode", "greedy"),
                                         data.get("allow_invalid", False))
                elif self.path == "/api/model":
                    result = viewer.load(data.get("checkpoint"))
                else:
                    self.reply(404, {"error": "Not found"})
                    return
                self.reply(200, result)
            except (ValueError, RuntimeError, OSError, BadZipFile, EOFError) as exc:
                self.reply(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return Handler


def export_replay(viewer, path, mode="greedy"):
    frames = [viewer.state()]
    while not viewer.finished:
        frames.append(viewer.step(mode=mode))
    payload = json.dumps(frames, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = (ASSETS / "index.html").read_text()
    html = html.replace('<link rel="stylesheet" href="/style.css">',
                        "<style>" + (ASSETS / "style.css").read_text() + "</style>")
    html = html.replace('<script src="/app.js" defer></script>',
                        "<script>window.INSPECTION_REPLAY=" + payload + ";</script>")
    html = html.replace("</body>", "<script>" + (ASSETS / "app.js").read_text() + "</script></body>")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--checkpoint", help="Optional .zip path inside models/")
    parser.add_argument("--export", metavar="HTML", help="Record a policy episode as a standalone HTML replay")
    parser.add_argument("--sampled", action="store_true", help="Sample actions for an exported replay")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.max_steps <= 1000:
        parser.error("Port must be 1..65535 and max steps must be 1..1000")
    if args.export and not args.checkpoint:
        parser.error("--export requires --checkpoint")
    torch.set_num_threads(1)
    viewer = Inspector(seed=args.seed, max_steps=args.max_steps)
    try:
        if args.checkpoint:
            viewer.load(args.checkpoint)
        if args.export:
            path = export_replay(viewer, args.export, "sampled" if args.sampled else "greedy")
            print(f"Recorded {viewer.env.steps} steps: {path}", flush=True)
            return
        with HTTPServer(("127.0.0.1", args.port), make_handler(viewer)) as server:
            print(f"Rescue inspector: http://127.0.0.1:{args.port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except PermissionError as exc:
        parser.exit(1, f"Permission denied: {exc}. For live inspection, run from your normal "
                       "terminal. For a replay, use --export with a writable HTML path.\n")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
