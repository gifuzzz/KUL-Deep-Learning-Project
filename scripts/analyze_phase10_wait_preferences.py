#!/usr/bin/env python3
"""Phase 10: Learned Wait Preferences Analysis (Section 8).

Analyzes action probabilities and logits from:
- Phase 6 Baseline
- Variant A (Control)
- Variant B (No Oscillation Penalty)
- Variant C (Repeated-Wait Penalty)
- Variant D (Combined Correction)

Evaluates on the exact deadlock decision states identified in Section 2:
1. Scenario 1: Dead end stall (Seed 50000, step 18+, where agent repeatedly selects Wait).
2. Scenario 2: Two-cell doorway oscillation (Seed 50057, step 25+, where agent oscillates between two doorway cells).
3. Scenario 3: Dead end escape decision point (Seed 50008, step 10+, junction with choice to backtrack or wait).

Computes:
- P(Wait) across models at critical decision points
- P(Movement actions: Up, Right, Down, Left)
- Entropy of action distribution
- Distribution shift: what replaces Wait when suppressed?
- Logit differences relative to Baseline and Control
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from rescue_env.envs.single_agent import RescueEnv
from rescue_env.wrappers import PreviousActionWrapper
from agents.recurrent_ppo import load_checkpoint, MaskableRecurrentPPO

ACTION_NAMES = [
    "Wait", "Move Up", "Move Right", "Move Down", "Move Left", "Pickup", "Water Hose"
]

MODELS = {
    "baseline": "models/phase6_spawn_curriculum/production_250k/stage3_spawn_curriculum/best_model.zip",
    "variant_a_control": "models/phase10_variant_a/final_model.zip",
    "variant_b_no_osc": "models/phase10_variant_b/final_model.zip",
    "variant_c_wait_penalty": "models/phase10_variant_c/final_model.zip",
    "variant_d_combined": "models/phase10_variant_d/final_model.zip",
}


def load_all_models(device: str = "cpu") -> dict[str, MaskableRecurrentPPO]:
    dummy_env = PreviousActionWrapper(RescueEnv(reward_shaping=False, geometry_mode="edge"))
    loaded = {}
    for name, path in MODELS.items():
        print(f"Loading {name} from {path}...")
        loaded[name] = load_checkpoint(path, env=dummy_env, device=device)
    dummy_env.close()
    return loaded


def evaluate_decision_point(
    env: Any,
    models: dict[str, MaskableRecurrentPPO],
    seed: int,
    task_type: str,
    tier: str | None,
    target_step: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Replays episode up to target_step, then evaluates all models on that exact state."""
    reset_opts = {}
    if task_type:
        reset_opts["task_type"] = task_type
    if tier:
        reset_opts["tier"] = tier

    # Rollout with baseline up to target_step to recreate the historical deadlock state
    base_model = models["baseline"]
    obs, info = env.reset(seed=seed, options=reset_opts)
    lstm_states = torch.zeros(1, 1, base_model.policy.hidden_size, device=device)
    episode_starts = torch.tensor([True], device=device)

    for step in range(target_step):
        action_mask = env.action_masks()
        obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
        mask_t = torch.as_tensor(np.array([action_mask], dtype=bool), device=device)

        with torch.no_grad():
            actions, _, _, lstm_states = base_model.policy.forward(
                obs_t,
                lstm_states=lstm_states,
                episode_starts=episode_starts,
                deterministic=True,
                action_masks=mask_t,
                return_hidden_state=True,
            )
            action = int(actions.item())
            episode_starts = torch.tensor([False], device=device)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Now at target state: evaluate all models
    current_pos = tuple(info.get("agent_pos", (0, 0)))
    action_mask = env.action_masks()
    obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
    mask_t = torch.as_tensor(np.array([action_mask], dtype=bool), device=device)

    state_analysis = {
        "step": target_step,
        "agent_pos": current_pos,
        "action_mask": [bool(m) for m in action_mask],
        "model_evaluations": {},
    }

    for m_name, model in models.items():
        with torch.no_grad():
            # Evaluation with clean zero hidden state or step hidden state
            # 1. With historical hidden state (for baseline) or fresh zero state (for others)
            # To be completely fair and comparable, evaluate both:
            # (a) Zero hidden state (pure state representation)
            # (b) Replay hidden state (replayed with this specific model from step 0)
            
            # Pure state representation
            h_zero = torch.zeros(1, 1, model.policy.hidden_size, device=device)
            ep_start = torch.tensor([True], device=device)
            dist_zero = model.policy.get_distribution(obs_t, lstm_states=h_zero, episode_starts=ep_start, action_masks=mask_t)
            logits_zero = dist_zero.distribution.logits.cpu().numpy()[0]
            probs_zero = dist_zero.distribution.probs.cpu().numpy()[0]

            entropy_zero = float(-np.sum([p * np.log(p + 1e-12) for p, m in zip(probs_zero, action_mask) if m]))
            
            p_wait = float(probs_zero[0])
            p_move = float(np.sum(probs_zero[1:5]))
            p_pickup = float(probs_zero[5])

            state_analysis["model_evaluations"][m_name] = {
                "p_wait": p_wait,
                "p_movement_total": p_move,
                "p_pickup": p_pickup,
                "probabilities": [float(p) for p in probs_zero],
                "logits": [float(l) for l in logits_zero],
                "entropy": entropy_zero,
                "argmax_action": int(np.argmax(logits_zero)),
                "argmax_action_name": ACTION_NAMES[int(np.argmax(logits_zero))],
            }

    return state_analysis


def run_full_preference_analysis(device: str = "cpu", output_file: str = "models/phase10_evaluations/phase10_wait_preferences.json"):
    models = load_all_models(device=device)

    env = PreviousActionWrapper(RescueEnv(
        task_type="exploration_only",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))

    results = {
        "deadlock_states": {},
        "summary": {},
    }

    # Point 1: Scenario 1 Dead End Stall (Seed 50000, exploration_only, step 18)
    print("\nEvaluating Deadlock Point 1: Dead End Stall (Seed 50000, step 18)...")
    results["deadlock_states"]["dead_end_stall_s50000_step18"] = evaluate_decision_point(
        env, models, seed=50000, task_type="exploration_only", tier=None, target_step=18, device=device
    )

    # Point 2: Scenario 1 Deep into Stall (Seed 50000, exploration_only, step 50)
    print("Evaluating Deadlock Point 2: Deep Stall (Seed 50000, step 50)...")
    results["deadlock_states"]["dead_end_deep_stall_s50000_step50"] = evaluate_decision_point(
        env, models, seed=50000, task_type="exploration_only", tier=None, target_step=50, device=device
    )

    env.close()

    # Point 3: Scenario 2 Doorway Oscillation (Seed 50057, full task, spawn_13_20, step 25)
    env = PreviousActionWrapper(RescueEnv(
        task_type="full",
        geometry_mode="edge",
        reward_shaping=False,
        distance_metric="spawn",
        spatial_memory_mode="none",
    ))
    print("Evaluating Deadlock Point 3: Doorway Oscillation (Seed 50057, step 25)...")
    results["deadlock_states"]["doorway_oscillation_s50057_step25"] = evaluate_decision_point(
        env, models, seed=50057, task_type="full", tier="spawn_13_20", target_step=25, device=device
    )

    # Point 4: Scenario 3 Legitimate Backtracking Decision Point (Seed 50008, full task, spawn_13_20, step 10)
    print("Evaluating Decision Point 4: Backtracking Junction (Seed 50008, step 10)...")
    results["deadlock_states"]["backtracking_junction_s50008_step10"] = evaluate_decision_point(
        env, models, seed=50008, task_type="full", tier="spawn_13_20", target_step=10, device=device
    )

    env.close()

    # Comparative Summary Table
    print("\n" + "=" * 95)
    print(f"{'State':32s} | {'Baseline':10s} | {'Var A (Ctrl)':12s} | {'Var B (NoOsc)':13s} | {'Var C (WaitPen)':15s} | {'Var D (Comb)':12s}")
    print("=" * 95)

    summary_table = {}
    for state_name, state_data in results["deadlock_states"].items():
        evals = state_data["model_evaluations"]
        row_str = f"{state_name:32s} | "
        row_dict = {}
        for m_name in ["baseline", "variant_a_control", "variant_b_no_osc", "variant_c_wait_penalty", "variant_d_combined"]:
            pw = evals[m_name]["p_wait"] * 100.0
            act = evals[m_name]["argmax_action_name"]
            row_str += f"{pw:4.1f}% ({act[:4]}) | "
            row_dict[m_name] = {
                "p_wait_pct": pw,
                "p_move_pct": evals[m_name]["p_movement_total"] * 100.0,
                "argmax_action": act,
                "entropy": evals[m_name]["entropy"],
            }
        print(row_str)
        summary_table[state_name] = row_dict
    print("=" * 95)

    results["summary"] = summary_table

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved wait preference analysis to {out_p}")


if __name__ == "__main__":
    run_full_preference_analysis(device="cpu")
