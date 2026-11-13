"""Train CNN actions from an observed-map teacher; the saved policy runs independently."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rescue_env.envs.observed_map import ObservedMap, DIRECTIONS
from rescue_env.envs.single_agent import RescueEnv


class ObservedTeacher:
    """Training labels from observations and known static grass, never hidden victims."""

    def __init__(self, height, width, margin=6):
        self.map = ObservedMap(height, width)
        self.margin = margin
        self.grass_added = False

    def add_grass(self):
        height, width, margin = self.map.height, self.map.width, self.margin
        grass = {(r, c) for r in range(-margin, height + margin)
                 for c in range(-margin, width + margin)
                 if not (0 <= r < height and 0 <= c < width)}
        self.map.known.update(grass)
        for cell in grass:
            for dr, dc in DIRECTIONS:
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in grass:
                    self.map.edges[cell].add(neighbor)
        self.grass_added = True

    def action(self, observation, mask):
        position = (round(float(observation['agent'][0]) * self.map.height),
                    round(float(observation['agent'][1]) * self.map.width))
        if not self.grass_added and not (0 <= position[0] < self.map.height
                                          and 0 <= position[1] < self.map.width):
            self.add_grass()
        step = self.map.observe(observation['grid'], position, bool(observation['agent'][-1]))
        if mask[5]:
            return 5
        delta = tuple(map(int, step[:2]))
        action = 0 if delta == (0, 0) else DIRECTIONS.index(delta) + 1
        if not mask[action]:
            raise RuntimeError(f'Teacher selected a blocked action at {position}: {action}')
        return action


class DemonstrationReplay:
    """Aggregate unique states across houses while retaining current learner corrections."""

    def __init__(self, capacity=8192):
        self.capacity = capacity
        self.examples = []
        self.recent = deque(maxlen=128)
        self.seen = {}
        self.conflicts = 0

    def add(self, observation, action, mask, rng):
        # shortcut: float16 images lose tiny hazard values; use float32 if hazard precision is needed.
        stored = {key: value.copy() if key == 'agent' else value.astype(np.float16)
                  for key, value in observation.items()}
        example = (stored, action, mask.copy())
        self.recent.append(example)
        digest = hashlib.blake2b(digest_size=16)
        for key in sorted(stored):
            digest.update(stored[key].tobytes())
        digest.update(mask.tobytes())
        signature = digest.digest()
        if signature in self.seen:
            self.conflicts += self.seen[signature] != action
            return
        self.seen[signature] = action
        if len(self.examples) < self.capacity:
            self.examples.append(example)
        else:
            index = rng.randrange(len(self.seen))
            if index < self.capacity:
                self.examples[index] = example

    def batch(self, rng, size=64):
        recent_count = min(16, len(self.recent))
        count = min(size - recent_count, len(self.examples))
        return rng.sample(self.examples, count) + rng.sample(list(self.recent), recent_count)


def imitate(policy, batch):
    observations = {key: torch.as_tensor(np.stack([row[0][key] for row in batch]),
                                       dtype=torch.float32, device=policy.device)
                    for key in batch[0][0]}
    actions = torch.as_tensor([row[1] for row in batch], device=policy.device)
    masks = np.stack([row[2] for row in batch])
    policy.set_training_mode(True)
    distribution = policy.get_distribution(observations, action_masks=masks)
    loss = -distribution.log_prob(actions).mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite imitation loss')
    policy.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
    policy.optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--teacher-mix-episodes', type=int, default=512)
    parser.add_argument('--student-stochastic', action='store_true')
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.episodes <= 8000 or args.teacher_mix_episodes < 0:
        parser.error('Choose a new output directory and 1–8000 episodes')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = MaskablePPO.load(args.model, device=args.device)
    model.set_random_seed(args.seed)
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
    near = RescueEnv(max_steps=64, victim_distance=4)
    far = RescueEnv(max_steps=1024, victim_distance=8, min_victim_distance=5)
    # Match evaluate_policy's per-environment episode counts so every validation root is used.
    counts = [(50 + i) // 8 for i in range(8)]
    partitions = []
    start = 10000
    for count in counts:
        partitions.append(list(range(start, start + count)))
        start += count
    validation = DummyVecEnv([
        lambda seeds=seeds: Monitor(RescueEnv(max_steps=1024, victim_distance=8,
                                             min_victim_distance=5, scenario_seeds=seeds))
        for seeds in partitions
    ])
    replay = DemonstrationReplay()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(dict(vars(args), output=str(args.output),
        teacher='ObservedTeacher: observed grid + own position + known static grass',
        training_roots=[1000, 1000 + args.episodes], validation_roots=[10000, 10050],
        max_far_steps=1024, replay_capacity=replay.capacity, learning_rate=1e-4,
        policy_observation_shapes={key: list(space.shape) for key, space in model.observation_space.spaces.items()}), indent=2))
    history = []
    best = -1
    steps = 0
    next_evaluation = 10000
    loss = 0.
    try:
        for episode in range(args.episodes):
            env = near if episode % 4 == 0 else far
            observation, _ = env.reset(seed=1000 + episode)
            footprint = observation['memory'][12]
            height = int((footprint.max(axis=1) > 0).sum())
            width = int((footprint.max(axis=0) > 0).sum())
            teacher = ObservedTeacher(height, width)
            done = False
            # Teacher rollouts first; then label learner-visited states (DAgger).
            teacher_probability = (max(0., 1. - episode / args.teacher_mix_episodes)
                                   if args.teacher_mix_episodes else 0.)
            while not done:
                mask = env.action_masks()
                expert = teacher.action(observation, mask)
                replay.add(observation, expert, mask, rng)
                if rng.random() < teacher_probability:
                    action = expert
                else:
                    action, _ = model.predict(observation, deterministic=not args.student_stochastic,
                                              action_masks=mask)
                    action = int(action)
                observation, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                model.imitation_steps = steps
                if replay.recent and steps % 32 == 0:
                    for _ in range(4):
                        loss = imitate(model.policy, replay.batch(rng))
                if steps >= next_evaluation:
                    rewards, lengths = evaluate_policy(model, validation, n_eval_episodes=50,
                        deterministic=True, return_episode_rewards=True, warn=False)
                    successes = sum(reward >= 10 for reward in rewards)
                    row = dict(steps=steps, episode=episode, validation_rescues=successes,
                               mean_length=float(np.mean(lengths)), loss=loss,
                               teacher_probability=teacher_probability,
                               unique_states=len(replay.seen), conflicting_labels=replay.conflicts)
                    history.append(row)
                    (args.output / 'evaluations.json').write_text(json.dumps(history, indent=2))
                    print(row, flush=True)
                    if successes > best:
                        best = successes
                        model.save(args.output / 'best_model')
                    model.save(args.output / 'last_model')
                    next_evaluation += 10000
                    if successes >= 48:
                        print('Validation threshold reached; freeze best for independent testing.', flush=True)
                        return
            if episode % 20 == 19:
                print(dict(episode=episode, steps=steps, mixed_rollout_success=info['success'],
                           profile='near' if env is near else 'far', imitation_loss=loss), flush=True)
        model.save(args.output / 'last_model')
    finally:
        near.close()
        far.close()
        validation.close()


if __name__ == '__main__':
    main()
