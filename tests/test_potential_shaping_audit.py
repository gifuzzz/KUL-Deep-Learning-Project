"""Mathematical and behavioral tests auditing potential-based reward shaping.

Verifies:
1. Two-state oscillation test: discounted returns under gamma=0.99 with terminal effects.
2. Continuous state potential across pickup: no negative penalty jump at pickup.
3. Clamp gradient preservation: max_distance >= 128 preserves non-zero gradient across building.
4. Terminal absorbing state potential is exactly 0.0.
"""

import pytest
import numpy as np
from rescue_env.envs.single_agent import RescueEnv
from rescue_env.envs.reward_shaping import (
    state_potential,
    potential_reward,
    shortest_distance,
    continuous_state_potential,
)


def test_two_state_oscillation_discounted_return():
    """Verify that under gamma=0.99, an oscillating trajectory has strictly finite discounted return

    and that completing the rescue yields a strictly higher discounted return.
    """
    gamma = 0.99
    beta = 0.1
    step_penalty = 0.001
    native_rescue_reward = 10.0
    initial_dist = 20.0
    max_steps = 384

    # 1. Complete discounted return of oscillating for 384 steps at distance 20:
    R_osc = 0.0
    for t in range(max_steps):
        if t % 2 == 0:
            b, a = -initial_dist, -(initial_dist + 1.0)
        else:
            b, a = -(initial_dist + 1.0), -initial_dist
        f = beta * (gamma * a - b) - step_penalty
        R_osc += (gamma ** t) * f

    # Telescoping check:
    # sum_{t=0}^{T-1} gamma^t (gamma Phi_{t+1} - Phi_t) = gamma^T Phi_T - Phi_0
    phi_0 = -initial_dist
    phi_T = -initial_dist  # end at same distance
    telescoping_potential = beta * ((gamma ** max_steps) * phi_T - phi_0)
    geometric_penalty = step_penalty * (1.0 - gamma ** max_steps) / (1.0 - gamma)
    R_osc_formula = telescoping_potential - geometric_penalty

    assert np.isclose(R_osc, R_osc_formula, atol=1e-5), f"Oscillation simulation {R_osc} != formula {R_osc_formula}"
    assert R_osc < 2.5, f"Oscillation return {R_osc} should be small and bounded"

    # 2. Complete discounted return of direct 20-step rescue:
    R_rescue = 0.0
    d = initial_dist
    for t in range(int(initial_dist)):
        b = -d
        if t == int(initial_dist) - 1:
            a = 0.0  # Terminal absorbing state
            step_native = native_rescue_reward
        else:
            d -= 1.0
            a = -d
            step_native = 0.0
        f = beta * (gamma * a - b) - step_penalty + step_native
        R_rescue += (gamma ** t) * f

    assert R_rescue > 10.0, f"Direct rescue return {R_rescue} should exceed native rescue reward"
    assert R_rescue > R_osc * 4.0, f"Rescue return {R_rescue} must dominate oscillation {R_osc}"


def test_pickup_potential_continuity():
    """Verify that continuous_state_potential eliminates the negative penalty jump at pickup."""
    env = RescueEnv(victim_distance=None, max_steps=384, navigation_shaping=True)
    obs, info = env.reset(seed=123)

    # Place agent directly at victim
    vr, vc = env.sim.victims[0]["r"], env.sim.victims[0]["c"]
    env.sim.agents[0]["r"] = vr
    env.sim.agents[0]["c"] = vc
    env.sim._core.set_people(env.sim.victims, env.sim.agents)
    env.sim.tick(0)

    # Measure before pickup
    before = continuous_state_potential(env.sim, max_distance=128)

    # Execute pickup (action 5)
    obs, r, term, trunc, info = env.step(5)

    # Measure after pickup
    after = continuous_state_potential(env.sim, max_distance=128)

    # Potential difference at pickup must be non-negative or negligible (no penalty)
    shaping_reward = potential_reward(before, after, gamma=0.99, beta=0.1)
    assert shaping_reward >= -0.05, f"Pickup shaping reward {shaping_reward} should not be a large negative penalty!"
    assert abs(after - before) <= 1.0, f"Potential jumped discontinuously from {before} to {after}"


def test_terminal_absorbing_state_potential():
    """Verify that terminal states have potential exactly 0.0."""
    env = RescueEnv(victim_distance=2, max_steps=64)
    obs, info = env.reset(seed=20000)

    # Dead agent potential is 0.0
    env.sim.agents[0]["dead"] = True
    assert continuous_state_potential(env.sim) == 0.0

    # Rescued victim potential is 0.0
    env.sim.agents[0]["dead"] = False
    env.sim.victims[0]["rescued"] = True
    assert continuous_state_potential(env.sim) == 0.0


def test_long_distance_gradient_not_clamped():
    """Verify that max_distance=128 preserves non-zero gradient for long-distance building paths."""
    env = RescueEnv(victim_distance=None, max_steps=384)
    env.reset(seed=10)
    agent = env.sim.agents[0]
    victim = env.sim.victims[0]
    vr, vc = victim["r"], victim["c"]

    path = env.sim.pathfind((agent["r"], agent["c"]), (vr, vc), world=True)
    assert len(path) > 2, "Path must have at least 2 steps"

    p0 = continuous_state_potential(env.sim, max_distance=128)
    # Move agent 1 step along path
    agent["r"], agent["c"] = path[1]
    env.sim._core.set_people(env.sim.victims, env.sim.agents)
    env.sim.tick(0)
    p1 = continuous_state_potential(env.sim, max_distance=128)

    diff = p1 - p0
    assert diff > 0.0, f"Potential gradient must be strictly positive when stepping closer, got diff={diff}"
