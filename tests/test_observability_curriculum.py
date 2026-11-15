"""Automated 12-Requirement Test Suite for Observability Curriculum (Phase 3).

Verifies the 12 strict requirements:
1. B1 full geometry + victim known
2. B2 full geometry + victim hidden
3. B3 partial geometry fractions (75%, 50%, 25%)
4. B4 observation equivalence to A
5. Partial-map unknown cells are impassable (0.0)
6. No false exploration bonuses on prefilled geometry
7. Shape invariance: (9, 15, 15), (9, 40, 52), (5,), ()
8. Action masks correct across stages
9. GRU resets and BPTT under curriculum conditions
10. Evaluation strictly unknown
11. Zero information leakage in A / B4
12. Backward compatibility with existing baseline checkpoints
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from rescue_env.envs.single_agent import (
    RescueEnv,
    extract_full_building_geometry,
    get_connected_revealed_mask,
)
from rescue_env.wrappers import PreviousActionWrapper
from agents.cnn import RecurrentSpatialRescueCNN
from agents.recurrent_ppo import MaskableRecurrentPPO
from training.train_ppo import (
    CurriculumRescueEnv,
    make_eval_env,
    make_train_env,
    load_checkpoint,
    set_env_observability_stage,
)


# =====================================================================
# Requirement 1: B1 Full Geometry + Victim Known
# =====================================================================
def test_requirement_1_b1_full_geometry_and_victim_known():
    """B1: 100% building geometry and victim location prefilled in memory map at t=0."""
    env = RescueEnv(
        geometry_reveal_fraction=1.0,
        reveal_victim=True,
        observability_stage="B1",
        victim_distance=10,
        min_victim_distance=5,
        max_steps=64,
    )
    obs, info = env.reset(seed=42)

    # 1. Full building geometry matches ground truth for all interior cells
    sim = env.unwrapped.sim
    full_geom = extract_full_building_geometry(sim)
    mem_grid = env.unwrapped.memory.grid
    interior = (sim.zone >= 0)

    # Check passability channels 0..3 and debris 4 on all interior cells
    np.testing.assert_allclose(
        mem_grid[0:5, interior],
        full_geom[0:5, interior],
        err_msg="Passability/debris do not match ground truth on interior cells",
    )

    # 2. Victim channel (6) has 1.0 at true victim location
    victim = sim.victims[0]
    vr, vc = victim["r"], victim["c"]
    assert mem_grid[6, vr, vc] == 1.0, f"Victim not marked at ({vr}, {vc}) in memory map"

    # 3. Explored mask (channel 5) was NOT artificially set to 1 by prefill
    # MemoryMap.explored must only track physical exploration (matches initial local line of sight, << interior cells)
    assert env.unwrapped.memory.explored.sum() == (obs["grid"][5] > 0).sum()
    assert env.unwrapped.memory.explored.sum() < interior.sum()
    env.close()


# =====================================================================
# Requirement 2: B2 Full Geometry + Victim Hidden
# =====================================================================
def test_requirement_2_b2_full_geometry_victim_hidden():
    """B2: 100% building geometry prefilled, but victim is hidden until physically observed."""
    env = RescueEnv(
        geometry_reveal_fraction=1.0,
        reveal_victim=False,
        observability_stage="B2",
        victim_distance=12,
        min_victim_distance=9,
        max_steps=64,
    )
    obs, info = env.reset(seed=42)

    sim = env.unwrapped.sim
    full_geom = extract_full_building_geometry(sim)
    mem_grid = env.unwrapped.memory.grid
    interior = (sim.zone >= 0)

    # Geometry is 100% known on all interior cells
    np.testing.assert_allclose(mem_grid[0:5, interior], full_geom[0:5, interior])

    # Victim is NOT known in memory map outside physical FOV
    victim = sim.victims[0]
    vr, vc = victim["r"], victim["c"]
    # Check that victim is outside initial local field of view
    agent = sim.agents[0]
    dist = abs(vr - agent["r"]) + abs(vc - agent["c"])
    assert dist > env.visibility_radius, f"Victim at ({vr},{vc}) too close to agent ({agent['r']},{agent['c']})"

    assert mem_grid[6, vr, vc] == 0.0, "Victim must remain hidden in B2"
    assert mem_grid[6].sum() == 0.0, "No victim marker should exist in memory in B2"
    env.close()


# =====================================================================
# Requirement 3: B3 Partial Geometry Fractions (75%, 50%, 25%)
# =====================================================================
@pytest.mark.parametrize("fraction", [0.75, 0.50, 0.25])
def test_requirement_3_b3_partial_geometry_fractions(fraction):
    """B3: Progressively hide geometry using contiguous flood-fill from agent start."""
    env = RescueEnv(
        geometry_reveal_fraction=fraction,
        reveal_victim=False,
        observability_stage=f"B3_{int(fraction*100)}",
        max_steps=64,
    )
    obs, info = env.reset(seed=42)

    sim = env.unwrapped.sim
    agent_pos = (sim.agents[0]["r"], sim.agents[0]["c"])
    revealed_mask = get_connected_revealed_mask(sim, start_pos=agent_pos, fraction=fraction)

    # Agent start position must be inside revealed mask
    assert revealed_mask[agent_pos[0], agent_pos[1]] == 1

    # Ratio of revealed reachable cells must closely track the fraction
    walkable_total = (sim.debris == 0).sum()
    revealed_count = (revealed_mask == 1).sum()
    actual_frac = revealed_count / walkable_total
    # Allow tolerance due to discrete room sizes and wall boundaries
    assert abs(actual_frac - fraction) < 0.20, f"Revealed fraction {actual_frac:.2f} too far from target {fraction}"

    # Verify that in memory, revealed cells have correct geometry and unrevealed cells are 0
    mem_grid = env.unwrapped.memory.grid
    full_geom = extract_full_building_geometry(sim)

    # For unrevealed cells, channels 0..4 must remain 0.0 (unless physically observed)
    unrevealed = (revealed_mask == 0) & (env.unwrapped.memory.explored == 0)
    if unrevealed.any():
        for ch in range(5):
            assert (mem_grid[ch][unrevealed] == 0.0).all(), f"Unrevealed cells in channel {ch} must be 0.0"

    env.close()


# =====================================================================
# Requirement 4: B4 Observation Equivalence to A
# =====================================================================
def test_requirement_4_b4_observation_equivalence_to_a():
    """B4 (fully unknown) produces bit-for-bit identical observations and transitions to A."""
    env_a = RescueEnv(
        geometry_reveal_fraction=0.0,
        reveal_victim=False,
        observability_stage="A",
        victim_distance=6,
        max_steps=64,
    )
    env_b4 = RescueEnv(
        geometry_reveal_fraction=0.0,
        reveal_victim=False,
        observability_stage="B4",
        victim_distance=6,
        max_steps=64,
    )

    obs_a, info_a = env_a.reset(seed=12345)
    obs_b4, info_b4 = env_b4.reset(seed=12345)

    np.testing.assert_array_equal(obs_a["grid"], obs_b4["grid"])
    np.testing.assert_array_equal(obs_a["memory"], obs_b4["memory"])
    np.testing.assert_array_equal(obs_a["agent"], obs_b4["agent"])

    # Step through 5 actions
    for action in [1, 2, 0, 4, 3]:
        o_a, r_a, term_a, trunc_a, i_a = env_a.step(action)
        o_b4, r_b4, term_b4, trunc_b4, i_b4 = env_b4.step(action)

        np.testing.assert_array_equal(o_a["grid"], o_b4["grid"])
        np.testing.assert_array_equal(o_a["memory"], o_b4["memory"])
        np.testing.assert_array_equal(o_a["agent"], o_b4["agent"])
        assert r_a == r_b4
        assert term_a == term_b4
        assert trunc_a == trunc_b4

    env_a.close()
    env_b4.close()


# =====================================================================
# Requirement 5: Partial-Map Unknown Cells are Impassable (0.0)
# =====================================================================
def test_requirement_5_partial_map_unknown_cells_impassable():
    """Unrevealed cells in partial observability (B3) must be 0.0 (impassable)."""
    env = RescueEnv(
        geometry_reveal_fraction=0.50,
        reveal_victim=False,
        observability_stage="B3b",
        victim_distance=8,
        max_steps=64,
    )
    obs, info = env.reset(seed=999)
    sim = env.unwrapped.sim
    revealed_mask = get_connected_revealed_mask(
        sim, start_pos=(sim.agents[0]["r"], sim.agents[0]["c"]), fraction=0.50
    )
    unrevealed = (revealed_mask == 0) & (env.unwrapped.memory.explored == 0)

    # Channels 0..3 are directional passabilities
    mem_grid = env.unwrapped.memory.grid
    for ch in range(4):
        assert (mem_grid[ch][unrevealed] == 0.0).all(), (
            f"Passability channel {ch} in unrevealed area must be 0.0 (impassable)"
        )
    env.close()


# =====================================================================
# Requirement 6: No False Exploration Bonuses on Prefilled Geometry
# =====================================================================
@pytest.mark.parametrize("obs_stage,frac,victim", [
    ("B1", 1.0, True),
    ("B2", 1.0, False),
    ("B3a", 0.75, False),
    ("B3b", 0.50, False),
    ("B3c", 0.25, False),
    ("B4", 0.0, False),
    ("A", 0.0, False),
])
def test_requirement_6_no_false_exploration_bonuses_on_prefill(obs_stage, frac, victim):
    """Prefilled memory map MUST NOT trigger false frontier exploration bonuses."""
    env = RescueEnv(
        geometry_reveal_fraction=frac,
        reveal_victim=victim,
        observability_stage=obs_stage,
        reward_shaping=True,
        max_steps=64,
    )
    obs, info = env.reset(seed=777)

    # Take STAY action (action 0). Agent doesn't move or discover new cells.
    obs, reward, terminated, truncated, info = env.step(0)

    # Reward must equal standard step penalty (-0.001), NOT containing exploration bonus (+0.05 * N)
    assert reward == pytest.approx(-0.001, abs=1e-5), (
        f"Stage {obs_stage} gave reward {reward} on STAY; exploration bonus leaked on prefill!"
    )
    assert info.get("total_exploration_reward", 0.0) == 0.0
    env.close()


# =====================================================================
# Requirement 7: Shape Invariance Across All Stages
# =====================================================================
@pytest.mark.parametrize("obs_stage,frac,victim", [
    ("B1", 1.0, True),
    ("B2", 1.0, False),
    ("B3a", 0.75, False),
    ("B3b", 0.50, False),
    ("B3c", 0.25, False),
    ("B4", 0.0, False),
    ("A", 0.0, False),
])
def test_requirement_7_shape_invariance(obs_stage, frac, victim):
    """Observation dictionary keys, shapes, and dtypes are strictly identical across all stages."""
    env = RescueEnv(
        geometry_reveal_fraction=frac,
        reveal_victim=victim,
        observability_stage=obs_stage,
    )
    env = PreviousActionWrapper(env)
    obs, _ = env.reset(seed=123)

    assert set(obs.keys()) == {"grid", "memory", "agent", "previous_action"}
    assert obs["grid"].shape == (9, 15, 15)
    assert obs["grid"].dtype == np.float32
    assert obs["memory"].shape == (9, 40, 52)
    assert obs["memory"].dtype == np.float32
    assert obs["agent"].shape == (5,)
    assert obs["agent"].dtype == np.float32
    assert obs["previous_action"].shape == ()
    assert obs["previous_action"].dtype == np.int64
    env.close()


# =====================================================================
# Requirement 8: Action Masks Correct Across Stages
# =====================================================================
def test_requirement_8_action_masks_correct():
    """Action masks strictly forbid moving into walls and picking up when no victim is present."""
    for obs_stage, frac in [("B1", 1.0), ("B2", 1.0), ("B3b", 0.5), ("B4", 0.0), ("A", 0.0)]:
        env = RescueEnv(
            geometry_reveal_fraction=frac,
            reveal_victim=(obs_stage == "B1"),
            observability_stage=obs_stage,
            victim_distance=8,
            min_victim_distance=4,
        )
        env = PreviousActionWrapper(env)
        obs, _ = env.reset(seed=42)
        masks = env.action_masks()

        assert len(masks) == 7
        assert masks.dtype == bool

        # Action 0 (STAY) is always legal
        assert masks[0] == True

        # Action 5 (PICKUP) must be False because victim is at distance 8
        assert masks[5] == False, f"Pickup was legal at start in stage {obs_stage}"

        # Raw actions mask from sim matches directional movement masks
        sim = env.unwrapped.sim
        raw = sim.observe(agent_id=0, visibility_radius=env.visibility_radius)
        np.testing.assert_array_equal(masks[1:5], raw["actions_mask"][1:5])
        env.close()


# =====================================================================
# Requirement 9: GRU Resets and Multi-Step BPTT
# =====================================================================
def test_requirement_9_gru_resets_and_bptt():
    """Verify recurrent policy BPTT gradient propagation and episode boundary hidden state resets."""
    from gymnasium import spaces
    obs_space = spaces.Dict({
        "grid": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 15, 15), dtype=np.float32),
        "memory": spaces.Box(low=-np.inf, high=np.inf, shape=(9, 40, 52), dtype=np.float32),
        "agent": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
        "previous_action": spaces.Box(low=0, high=6, shape=(), dtype=np.int64),
    })
    action_space = spaces.Discrete(7)
    policy = RecurrentSpatialRescueCNN(obs_space, action_space)
    policy.set_training_mode(True)

    # 3 consecutive timesteps for 1 sequence: batch shape (3, ...)
    g0 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g1 = torch.randn(1, 9, 15, 15, requires_grad=True)
    g2 = torch.randn(1, 9, 15, 15, requires_grad=True)
    batch_grid = torch.cat([g0, g1, g2], dim=0)

    batch_obs = {
        "grid": batch_grid,
        "memory": torch.randn(3, 9, 40, 52),
        "agent": torch.randn(3, 5),
        "previous_action": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    actions = torch.tensor([1, 2, 3], dtype=torch.long)
    masks = torch.ones((3, 7), dtype=torch.bool)
    h0 = torch.zeros((1, 1, 128))
    # Episode starts at t=0, continues at t=1, continues at t=2
    episode_starts = torch.tensor([1.0, 0.0, 0.0])

    values, log_prob, entropy = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts,
        action_masks=masks,
    )

    # Loss computed solely at t=2
    loss_t2 = values[2] + log_prob[2]
    grad_g0 = torch.autograd.grad(loss_t2, g0, retain_graph=True)[0]
    assert grad_g0 is not None
    assert grad_g0.abs().sum().item() > 0.0, "Multi-step BPTT: gradient dL_2 / ds_0 must be non-zero"

    # Reset isolation: if episode resets at t=1, gradient into t=0 is zero
    episode_starts_reset = torch.tensor([1.0, 1.0, 0.0])
    values_r, log_prob_r, _ = policy.evaluate_actions(
        batch_obs,
        actions,
        lstm_states=h0,
        episode_starts=episode_starts_reset,
        action_masks=masks,
    )
    loss_t2_r = values_r[2] + log_prob_r[2]
    grad_g0_r = torch.autograd.grad(loss_t2_r, g0, retain_graph=True)[0]
    assert grad_g0_r.abs().sum().item() == 0.0, "Reset isolation: episode start at t=1 must sever gradient to t=0"


# =====================================================================
# Requirement 10: Evaluation Strictly Unknown
# =====================================================================
def test_requirement_10_evaluation_strictly_unknown():
    """All evaluation environments MUST run under fully unknown and unshaped conditions."""
    eval_env = make_eval_env(victim_distance=12, max_steps=128, min_victim_distance=9)

    # Unwrap past PreviousActionWrapper to inspect RescueEnv
    inner = eval_env.unwrapped
    assert inner.geometry_reveal_fraction == 0.0, "Evaluation geometry_reveal_fraction must be 0.0"
    assert inner.reveal_victim is False, "Evaluation reveal_victim must be False"
    assert inner.reward_shaping is False, "Evaluation reward_shaping must be False"
    assert inner.navigation_shaping is False, "Evaluation navigation_shaping must be False"

    # Reset and check observation at t=0
    obs, _ = eval_env.reset(seed=42)
    assert obs["memory"][6].sum() == 0.0, "Victim must never be prefilled in eval memory map"
    assert obs["grid"][6].sum() == 0.0, "Distant victim must not be visible in local grid"

    # Walls outside initial line of sight must be 0.0
    explored_cells = (obs["memory"][5] > 0)
    unexplored_cells = ~explored_cells
    assert (obs["memory"][0:4][:, unexplored_cells] == 0.0).all(), (
        "Unexplored walls must not appear in evaluation memory map"
    )
    eval_env.close()


# =====================================================================
# Requirement 11: Zero Information Leakage in A and B4
# =====================================================================
def test_requirement_11_zero_information_leakage_in_a_and_b4():
    """In A and B4, verify absolute zero privileged simulator information leakage."""
    for obs_stage in ["A", "B4"]:
        env = RescueEnv(
            geometry_reveal_fraction=0.0,
            reveal_victim=False,
            observability_stage=obs_stage,
            victim_distance=12,
            min_victim_distance=8,
            max_steps=64,
        )
        obs, _ = env.reset(seed=54321)

        sim = env.unwrapped.sim
        vr, vc = sim.victims[0]["r"], sim.victims[0]["c"]

        # Ensure victim is outside initial visibility radius
        ar, ac = sim.agents[0]["r"], sim.agents[0]["c"]
        assert abs(vr - ar) + abs(vc - ac) > env.visibility_radius

        # Check local grid victim channel
        assert obs["grid"][6].sum() == 0.0, f"Victim leaked into local grid in stage {obs_stage}"

        # Check memory victim channel
        assert obs["memory"][6].sum() == 0.0, f"Victim leaked into memory map in stage {obs_stage}"

        # Check that agent scalar vector contains ONLY [r_norm, c_norm, hp, alive, carrying]
        # and has zero compass / angle / distance to victim
        assert obs["agent"].shape == (5,)
        # Values 0,1 are normalized r,c; 2 is hp; 3 is alive; 4 is carrying (0.0)
        assert obs["agent"][4] == 0.0

        env.close()


# =====================================================================
# Requirement 12: Backward Compatibility with Baseline Checkpoints
# =====================================================================
def test_requirement_12_backward_compatibility_with_baseline_checkpoints():
    """Existing baseline checkpoints load cleanly and produce valid actions in the environment."""
    ckpt_path = Path("models/architecture_ablation/baseline/stage2c/best_model.zip")
    if not ckpt_path.exists():
        ckpt_path = Path("models/curriculum_stage2/stage2/stage2_final.zip")

    if not ckpt_path.exists():
        pytest.skip(f"No baseline checkpoint found at {ckpt_path}")

    dummy_env = make_eval_env(victim_distance=2, max_steps=64)
    model = load_checkpoint(ckpt_path, env=dummy_env, device="cpu")

    obs, _ = dummy_env.reset(seed=42)
    masks = dummy_env.action_masks()

    action, _ = model.predict(obs, deterministic=True, action_masks=masks)
    act_int = int(action)

    assert 0 <= act_int < 7
    assert masks[act_int] == True, f"Policy selected illegal action {act_int} under mask {masks}"

    dummy_env.close()
