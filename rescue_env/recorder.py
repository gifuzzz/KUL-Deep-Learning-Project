"""Trajectory and Episode Recorder for FlameBox Visual Debugger.

Captures ground-truth simulation snapshots, exact policy-input tensors,
neural decision outputs (raw logits, masked probabilities, values, GRU norm),
step transitions, failure events, and categorization for deterministic replay.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ACTION_NAMES = (
    "Wait",
    "Move Up",
    "Move Right",
    "Move Down",
    "Move Left",
    "Pick Up / Drop",
    "Water Hose",
)

FEATURE_NAMES = (
    "Row / Height",
    "Column / Width",
    "Health / 100",
    "Role / 3",
    "Carrying Victim",
)

MEMORY_CHANNELS = (
    ("Passability North", "Traversable northward edge (flux >= 0.5)", "#388f8d"),
    ("Passability East", "Traversable eastward edge (flux >= 0.5)", "#388f8d"),
    ("Passability South", "Traversable southward edge (flux >= 0.5)", "#388f8d"),
    ("Passability West", "Traversable westward edge (flux >= 0.5)", "#388f8d"),
    ("Debris / Obstacle", "Blocked by rubble or structural collapse", "#b08763"),
    ("Explored / Visited", "Observed inside line-of-sight", "#668fbb"),
    ("Victim", "Known victim presence and category", "#bc6d98"),
    ("Exit", "Exterior door / window exit point", "#36996a"),
    ("Agent Position", "Firefighter location in memory map", "#e27b35"),
)


def compute_model_hash(model_path: str | Path | None) -> str:
    """Compute sha256 checksum of a checkpoint file."""
    if model_path is None:
        return "unknown"
    p = Path(model_path)
    if not p.exists():
        return "not_found"
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "error"


def snapshot_ground_truth(env: Any) -> dict[str, Any]:
    """Capture current simulation ground truth state before environment step."""
    sim = getattr(env, "sim", None)
    person = sim.agents[0] if sim and len(sim.agents) > 0 else {"r": 0, "c": 0, "health": 100}
    victims_data = []
    carrying = False
    if sim and hasattr(sim, "victims"):
        for idx, v in enumerate(sim.victims):
            is_carried = v.get("carried_by", -1) == 0
            if is_carried and not v.get("dead", False) and not v.get("rescued", False):
                carrying = True
            victims_data.append({
                "id": idx,
                "r": int(v["r"]),
                "c": int(v["c"]),
                "carried_by": int(v.get("carried_by", -1)),
                "rescued": bool(v.get("rescued", False)),
                "dead": bool(v.get("dead", False)),
            })

    debris_list = []
    if sim and hasattr(sim, "debris"):
        dr, dc = np.where(sim.debris > 0)
        debris_list = [[int(r), int(c)] for r, c in zip(dr, dc)]

    fire_list = []
    if sim and hasattr(sim, "fire"):
        fr, fc = np.where(sim.fire > 0)
        for r, c in zip(fr, fc):
            fire_list.append([int(r), int(c), float(sim.fire[r, c])])

    smoke_list = []
    if sim and hasattr(sim, "smoke"):
        sr, sc = np.where(sim.smoke > 0)
        for r, c in zip(sr, sc):
            smoke_list.append([int(r), int(c), float(sim.smoke[r, c])])

    return {
        "agent_pos": [int(person["r"]), int(person["c"])],
        "agent_health": float(person.get("health", 100)),
        "carrying": bool(carrying),
        "victims": victims_data,
        "debris": debris_list,
        "fire": fire_list,
        "smoke": smoke_list,
    }


class EpisodeRecorder:
    """Records an entire FlameBox episode with ground truth and exact policy tensors."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}
        self.static_ground_truth: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.is_finished: bool = False

    def start_episode(
        self,
        env: Any,
        seed: int,
        deterministic: bool = True,
        model_path: str | Path | None = None,
        architecture: str = "baseline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize recording at episode start."""
        self.metadata = {
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "model_path": str(model_path) if model_path else None,
            "model_hash": compute_model_hash(model_path),
            "architecture": str(architecture),
            "max_steps": getattr(env, "max_steps", 500),
            "victim_distance": getattr(env, "victim_distance", None),
            "min_victim_distance": getattr(env, "min_victim_distance", 1),
            "task_type": getattr(env, "task_type", "full"),
            "options": options or {},
            "geometry_mode": getattr(env, "geometry_mode", "edge"),
            "version": "2.0.0",
        }

        # Determine difficulty tier name
        v_dist = getattr(env, "victim_distance", None)
        if v_dist == 2:
            self.metadata["tier"] = "Near"
        elif v_dist == 6:
            self.metadata["tier"] = "Medium"
        elif v_dist == 8:
            self.metadata["tier"] = "Bridge A"
        elif v_dist == 10:
            self.metadata["tier"] = "Bridge B"
        elif v_dist == 12:
            self.metadata["tier"] = "Bridge C"
        elif v_dist is None:
            self.metadata["tier"] = "Distant"
        else:
            self.metadata["tier"] = f"d={v_dist}"

        # Extract static building ground truth from env.sim
        sim = getattr(env, "sim", None)
        if sim is not None:
            openings = []
            exits = []
            for op in getattr(sim, "openings", []):
                portal_cells = []
                if hasattr(sim, "_portal_cells") and "zone_a" in op:
                    try:
                        portal_cells = [[int(r), int(c)] for r, c in sim._portal_cells(op, op["zone_a"])]
                    except Exception:
                        portal_cells = []
                za = op.get("zone_a")
                zb = op.get("zone_b")
                zone_a_val = int(za) if za is not None else -1
                zone_b_val = int(zb) if zb is not None else -1

                openings.append({
                    "id": int(op.get("id", len(openings))),
                    "kind": str(op.get("kind", "door")),
                    "orient": str(op.get("orient", "H")),
                    "at": int(op.get("at", 0)),
                    "start": int(op.get("start", 0)),
                    "length": int(op.get("length", 1)),
                    "open": bool(op.get("open", True)),
                    "blocked": bool(op.get("blocked", False)),
                    "zone_a": zone_a_val,
                    "zone_b": zone_b_val,
                    "cells": portal_cells,
                })
                if zone_b_val == -1:
                    for cell in portal_cells:
                        if 0 <= cell[0] < sim.h and 0 <= cell[1] < sim.w:
                            exits.append(cell)

            self.static_ground_truth = {
                "height": int(sim.h),
                "width": int(sim.w),
                "zones": sim.zone.tolist() if hasattr(sim, "zone") else [],
                "south": (sim.south >= 0.5).astype(int).tolist() if hasattr(sim, "south") else [],
                "east": (sim.east >= 0.5).astype(int).tolist() if hasattr(sim, "east") else [],
                "openings": openings,
                "exits": exits,
                "geometry_mode": getattr(env, "geometry_mode", "edge"),
                "max_height": int(getattr(env, "MEMORY_HEIGHT", 40)),
                "max_width": int(getattr(env, "MEMORY_WIDTH", 52)),
                "pad_top": int((getattr(env, "MEMORY_HEIGHT", 40) - sim.h) // 2),
                "pad_left": int((getattr(env, "MEMORY_WIDTH", 52) - sim.w) // 2),
            }
        else:
            self.static_ground_truth = {}

        self.steps = []
        self.events = []
        self.summary = {}
        self.is_finished = False

    def record_step(
        self,
        step_idx: int,
        env: Any,
        obs: dict[str, np.ndarray | float],
        action_mask: np.ndarray | list[bool],
        raw_logits: np.ndarray | torch.Tensor,
        masked_logits: np.ndarray | torch.Tensor,
        action_probs: np.ndarray | torch.Tensor,
        selected_action: int,
        value_estimate: float,
        gru_hidden_norm: float,
        reward: float,
        cumulative_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        ground_truth_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Record a single execution step."""
        gt_data = ground_truth_snapshot if ground_truth_snapshot is not None else snapshot_ground_truth(env)

        # Convert tensors to python lists
        def _to_list(arr):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy().tolist()
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return list(arr)

        raw_logits_list = _to_list(raw_logits)
        if len(raw_logits_list) > 0 and isinstance(raw_logits_list[0], list):
            raw_logits_list = raw_logits_list[0]

        masked_logits_list = _to_list(masked_logits)
        if len(masked_logits_list) > 0 and isinstance(masked_logits_list[0], list):
            masked_logits_list = masked_logits_list[0]

        probs_list = _to_list(action_probs)
        if len(probs_list) > 0 and isinstance(probs_list[0], list):
            probs_list = probs_list[0]

        # Policy inputs exact capture
        grid_arr = obs["grid"]
        if isinstance(grid_arr, torch.Tensor):
            grid_arr = grid_arr.detach().cpu().numpy()
        grid_list = grid_arr.astype(np.float32).tolist()

        mem_arr = obs["memory"]
        if isinstance(mem_arr, torch.Tensor):
            mem_arr = mem_arr.detach().cpu().numpy()
        mem_list = mem_arr.astype(np.float32).tolist()

        agent_arr = obs["agent"]
        if isinstance(agent_arr, torch.Tensor):
            agent_arr = agent_arr.detach().cpu().numpy()
        agent_list = agent_arr.astype(np.float32).tolist()

        prev_act = obs.get("previous_action", 0)
        if isinstance(prev_act, (torch.Tensor, np.ndarray)):
            prev_act = int(prev_act.item())
        else:
            prev_act = int(prev_act)

        act_mask_list = [bool(m) for m in action_mask]

        step_dict = {
            "step": int(step_idx),
            "ground_truth": gt_data,
            "policy_inputs": {
                "grid": grid_list,
                "memory": mem_list,
                "agent": agent_list,
                "previous_action": prev_act,
                "action_mask": act_mask_list,
            },
            "policy_outputs": {
                "raw_logits": [round(float(x), 4) for x in raw_logits_list],
                "masked_logits": [round(float(x), 4) for x in masked_logits_list],
                "action_probs": [round(float(x), 5) for x in probs_list],
                "selected_action": int(selected_action),
                "action_name": ACTION_NAMES[selected_action] if 0 <= selected_action < len(ACTION_NAMES) else str(selected_action),
                "value_estimate": round(float(value_estimate), 4),
                "gru_hidden_norm": round(float(gru_hidden_norm), 4),
            },
            "transition": {
                "reward": float(reward),
                "cumulative_reward": round(float(cumulative_reward), 4),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "done": bool(terminated or truncated),
                "timeout": bool(info.get("timeout", False)),
                "victim_seen": bool(info.get("victim_seen", False)),
                "rescue_completed": bool(reward >= 9.9 or info.get("rescued", False)),
            },
        }

        self.steps.append(step_dict)

    def finish_episode(self, env: Any | None = None) -> None:
        """Compute episode statistics, failure events, and categorization."""
        if self.is_finished or not self.steps:
            return

        total_steps = len(self.steps)
        total_reward = self.steps[-1]["transition"]["cumulative_reward"]
        final_transition = self.steps[-1]["transition"]
        rescue_completed = any(s["transition"]["rescue_completed"] for s in self.steps)
        timeout = final_transition["timeout"] or final_transition["truncated"]

        # Track discovery and pickup steps
        discovery_step = None
        pickup_step = None
        rescue_step = None

        carrying_states = [s["ground_truth"]["carrying"] for s in self.steps]
        agent_positions = [tuple(s["ground_truth"]["agent_pos"]) for s in self.steps]

        for s in self.steps:
            step_idx = s["step"]
            # Discovery: when channel 6 (Victim) has value > 0 in local grid
            # In Phase 2 contract, local_grid[6] is victim channel
            grid_ch6 = np.array(s["policy_inputs"]["grid"][6])
            if discovery_step is None and (grid_ch6 > 0).any():
                discovery_step = step_idx
                self.events.append({
                    "step": step_idx,
                    "type": "first_discovery",
                    "description": "Victim first entered line-of-sight",
                    "r": s["ground_truth"]["agent_pos"][0],
                    "c": s["ground_truth"]["agent_pos"][1],
                })

            if pickup_step is None and s["ground_truth"]["carrying"]:
                pickup_step = step_idx
                self.events.append({
                    "step": step_idx,
                    "type": "first_pickup",
                    "description": "Firefighter picked up victim",
                    "r": s["ground_truth"]["agent_pos"][0],
                    "c": s["ground_truth"]["agent_pos"][1],
                })

            if s["transition"]["rescue_completed"] and rescue_step is None:
                rescue_step = step_idx
                self.events.append({
                    "step": step_idx,
                    "type": "rescue_completed",
                    "description": f"Victim successfully evacuated outside (+10 rescue reward) at step {step_idx}",
                    "r": s["ground_truth"]["agent_pos"][0],
                    "c": s["ground_truth"]["agent_pos"][1],
                })

        # Event: Two-cell movement oscillation detection (A -> B -> A -> B)
        oscillation_count = 0
        osc_events_added = set()
        for i in range(len(agent_positions) - 3):
            p0, p1, p2, p3 = agent_positions[i : i + 4]
            if p0 == p2 and p1 == p3 and p0 != p1:
                oscillation_count += 1
                if i not in osc_events_added and (i - 1) not in osc_events_added:
                    osc_events_added.add(i)
                    self.events.append({
                        "step": i + 2,
                        "type": "two_cell_oscillation",
                        "description": f"2-cell oscillation between {p0} and {p1}",
                        "r": p2[0],
                        "c": p2[1],
                    })

        # Event: Repeated unsuccessful action (same action repeated >= 3 times with no position change)
        for i in range(len(self.steps) - 2):
            a0 = self.steps[i]["policy_outputs"]["selected_action"]
            a1 = self.steps[i + 1]["policy_outputs"]["selected_action"]
            a2 = self.steps[i + 2]["policy_outputs"]["selected_action"]
            p0 = agent_positions[i]
            p1 = agent_positions[i + 1]
            p2 = agent_positions[i + 2]
            if a0 == a1 == a2 and p0 == p1 == p2 and a0 in (1, 2, 3, 4, 5):
                self.events.append({
                    "step": i + 2,
                    "type": "repeated_unsuccessful",
                    "description": f"Repeated action '{ACTION_NAMES[a0]}' 3 times without moving",
                    "r": p2[0],
                    "c": p2[1],
                })
                break  # Don't flood with duplicate repeated events

        # Event: Room transition (zone changes)
        if self.static_ground_truth.get("zones"):
            zones_map = np.array(self.static_ground_truth["zones"])
            h, w = zones_map.shape
            last_zone = None
            for i, (r, c) in enumerate(agent_positions):
                cur_zone = int(zones_map[r, c]) if (0 <= r < h and 0 <= c < w) else -1
                if last_zone is not None and cur_zone != last_zone:
                    self.events.append({
                        "step": i,
                        "type": "room_transition",
                        "description": f"Transitioned from Zone {last_zone} to Zone {cur_zone}",
                        "r": r,
                        "c": c,
                    })
                last_zone = cur_zone

        # Event: Reached known exit
        exits = self.static_ground_truth.get("exits", [])
        for i, pos in enumerate(agent_positions):
            if list(pos) in exits and carrying_states[i]:
                self.events.append({
                    "step": i,
                    "type": "exit_reached",
                    "description": f"Reached exit while carrying victim at {pos}",
                    "r": pos[0],
                    "c": pos[1],
                })
                break

        # Event: Timeout
        if timeout:
            final_pos = agent_positions[-1]
            self.events.append({
                "step": total_steps - 1,
                "type": "timeout",
                "description": f"Episode reached maximum step budget ({total_steps} steps)",
                "r": final_pos[0],
                "c": final_pos[1],
            })

        # Explored-map coverage
        final_mem = np.array(self.steps[-1]["policy_inputs"]["memory"])
        explored_count = int(np.count_nonzero(final_mem[5] > 0))
        total_interior = int(np.count_nonzero(np.array(self.static_ground_truth.get("zones", [])) >= 0)) if self.static_ground_truth.get("zones") else 1
        coverage_fraction = round(explored_count / max(1, total_interior), 4)

        # Categorize episode
        ever_picked_up = bool(any(carrying_states))
        tier = self.metadata.get("tier", "Unknown")

        if rescue_completed:
            if tier == "Near":
                category = "successful_near_rescue"
            elif tier == "Medium":
                category = "successful_medium_rescue"
            elif tier in ("Bridge A", "Bridge B", "Bridge C"):
                category = "successful_long_distance_rescue"
            elif tier == "Distant":
                category = "successful_distant_rescue"
            else:
                category = "successful_rescue"
        else:
            if ever_picked_up:
                # Picked up but not returned
                if oscillation_count >= 3:
                    category = "doorway_oscillation"
                else:
                    category = "carrier_timeout"
            else:
                # Never picked up
                if discovery_step is not None:
                    category = "discovered_not_picked"
                else:
                    if oscillation_count >= 5:
                        category = "doorway_oscillation"
                    elif coverage_fraction > 0.6:
                        category = "cell_revisits"
                    else:
                        category = "exploration_timeout"

        self.summary = {
            "total_steps": total_steps,
            "total_reward": round(total_reward, 4),
            "rescue_completed": rescue_completed,
            "timeout": timeout,
            "discovery_step": discovery_step,
            "pickup_step": pickup_step,
            "rescue_step": rescue_step,
            "ever_picked_up": ever_picked_up,
            "carrying_at_end": carrying_states[-1] if carrying_states else False,
            "explored_coverage": coverage_fraction,
            "oscillation_count": oscillation_count,
            "category": category,
            "category_readable": category.replace("_", " ").title(),
        }

        # Sort events chronologically
        self.events.sort(key=lambda e: e["step"])
        self.is_finished = True

    def to_dict(self) -> dict[str, Any]:
        """Convert entire recording to a JSON-serializable dictionary."""
        return {
            "metadata": self.metadata,
            "static_ground_truth": self.static_ground_truth,
            "summary": self.summary,
            "events": self.events,
            "steps": self.steps,
        }

    def save(self, filepath: str | Path) -> Path:
        """Save episode trajectory to JSON or compressed JSON (.json.gz)."""
        self.finish_episode()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        if str(path).endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        return path

    @classmethod
    def load(cls, filepath: str | Path) -> EpisodeRecorder:
        """Load an episode trajectory from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        rec = cls()
        rec.metadata = data.get("metadata", {})
        rec.static_ground_truth = data.get("static_ground_truth", {})
        rec.summary = data.get("summary", {})
        rec.events = data.get("events", [])
        rec.steps = data.get("steps", [])
        rec.is_finished = True
        return rec


def record_episode(
    model: Any,
    env: Any,
    seed: int,
    deterministic: bool = True,
    model_path: str | Path | None = None,
    architecture: str = "baseline",
    options: dict[str, Any] | None = None,
) -> EpisodeRecorder:
    """Execute and faithfully record an episode with exact policy-input and output tensors."""
    # Initialize environment first so env.sim is populated
    obs, info = env.reset(seed=seed, options=options)
    action_mask = env.action_masks()

    recorder = EpisodeRecorder()
    recorder.start_episode(
        env=env,
        seed=seed,
        deterministic=deterministic,
        model_path=model_path,
        architecture=architecture,
        options=options,
    )

    # Recurrent state tracking
    gru_states = None
    episode_starts = np.ones((1,), dtype=bool)

    done = False
    step_idx = 0
    total_reward = 0.0

    target_device = model.device if hasattr(model, "device") else torch.device("cpu")

    while not done:
        # 1. Convert current observation into tensor
        obs_tensor, _ = model.policy.obs_to_tensor(obs)

        # 2. Convert state and episode start tensors
        st_tensor = None
        if gru_states is not None:
            if isinstance(gru_states, tuple):
                st_tensor = torch.as_tensor(gru_states[0], dtype=torch.float32, device=target_device)
            elif isinstance(gru_states, np.ndarray):
                st_tensor = torch.as_tensor(gru_states, dtype=torch.float32, device=target_device)
            elif isinstance(gru_states, torch.Tensor):
                st_tensor = gru_states.to(target_device)

        ep_start_tensor = torch.as_tensor(episode_starts, dtype=torch.float32, device=target_device)

        # 3. Exact policy inference capturing intermediate diagnostics
        with torch.no_grad():
            latent, next_h = model.policy._forward_network(
                obs_tensor,
                lstm_states=st_tensor,
                episode_starts=ep_start_tensor,
            )
            raw_logits = model.policy.action_net(latent)
            values = model.policy.value_net(latent)
            dist = model.policy.action_dist.proba_distribution(action_logits=raw_logits)
            dist.apply_masking(action_mask)
            masked_logits = dist.distribution.logits
            probs = dist.distribution.probs

            if deterministic:
                action = int(torch.argmax(masked_logits, dim=-1).item())
            else:
                action = int(dist.sample().item())

            value_val = float(values.squeeze().item())
            gru_norm = float(torch.norm(next_h, p=2).item())

        # 4. Snapshot simulation ground truth BEFORE taking action
        gt_snapshot = snapshot_ground_truth(env)

        # 5. Step environment with selected action
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        # 6. Record step data
        recorder.record_step(
            step_idx=step_idx,
            env=env,
            obs=obs,
            action_mask=action_mask,
            raw_logits=raw_logits,
            masked_logits=masked_logits,
            action_probs=probs,
            selected_action=action,
            value_estimate=value_val,
            gru_hidden_norm=gru_norm,
            reward=reward,
            cumulative_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
            ground_truth_snapshot=gt_snapshot,
        )

        # 6. Advance recurrent state and environment observations
        obs = next_obs
        action_mask = env.action_masks()
        gru_states = next_h.cpu().numpy()
        episode_starts = np.zeros((1,), dtype=bool)
        done = terminated or truncated
        step_idx += 1

    recorder.finish_episode(env=env)
    return recorder
