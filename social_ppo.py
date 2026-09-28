"""Tabular PPO with an adaptive social imitation extension."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from gridworld import DELTAS, HEIGHT, WIDTH, Scenario, distance_map, move, state_id


@dataclass
class Config:
    episodes: int = 500
    max_steps: int = 100
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    update_epochs: int = 4
    actor_lr: float = 0.04
    critic_lr: float = 0.06
    entropy_coef: float = 0.01
    demo_episodes: int = 10
    stationary_window: int = 3
    jaccard_threshold: float = 0.1
    jaccard_decay: float = 5.0


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def expert_trajectory(start: tuple[int, int], goal: tuple[int, int],
                      rng: np.random.Generator, stationary_window: int,
                      blocked: frozenset[tuple[int, int]] = frozenset()):
    """Shortest path with random tie breaking, followed by a stationary tail."""
    position = start
    states = [position]
    actions = []
    distances = distance_map(goal, blocked)
    while position != goal:
        candidates = [action for action in range(4)
                      if distances.get(move(position, action, blocked), float("inf"))
                      < distances[position]]
        action = int(rng.choice(candidates))
        actions.append(action)
        position = move(position, action, blocked)
        states.append(position)
    states.extend([goal] * stationary_window)
    return states, actions


def select_expert_and_lookup(scenario: Scenario, goal: tuple[int, int],
                             rng: np.random.Generator, cfg: Config,
                             trajectories=None, forced_expert: int | None = None):
    """Vote over ten episodes for an expert stationary at the learner's goal."""
    starts_and_goals = (
        (scenario.red_expert_start, scenario.red_goal),
        (scenario.blue_expert_start, scenario.blue_goal),
    )
    if trajectories is None:
        trajectories = [
            [expert_trajectory(start, target, rng, cfg.stationary_window, scenario.blocked)
             for _ in range(cfg.demo_episodes)]
            for start, target in starts_and_goals
        ]
    votes = []
    for expert_traces in trajectories:
        votes.append(sum(
            any(len(set(states[i:i + cfg.stationary_window])) == 1
                and states[i] == goal
                for i in range(len(states) - cfg.stationary_window + 1))
            for states, _ in expert_traces
        ))
    winner = int(np.argmax(votes)) if forced_expert is None else forced_expert
    if forced_expert is None and votes[winner] == 0:
        return None, {}, votes

    counts = {}
    for states, actions in trajectories[winner]:
        for position, action in zip(states, actions):
            counts.setdefault(state_id(position), Counter())[action] += 1
    lookup = {state: count.most_common(1)[0][0] for state, count in counts.items()}
    return winner, lookup, votes


def adaptive_coefficient(rollout_states: np.ndarray, lookup: dict[int, int],
                         cfg: Config) -> float:
    """Jaccard(S_exp, S_rollout), with S_exp the matched rollout states."""
    visited = set(map(int, rollout_states))
    matched = visited.intersection(lookup)
    jaccard = len(matched) / len(visited) if visited else 0.0
    if jaccard < cfg.jaccard_threshold:
        return float(jaccard * np.exp(-cfg.jaccard_decay *
                                      (cfg.jaccard_threshold - jaccard)))
    return float(jaccard)


class TabularPPO:
    """Clipped PPO actor with a tabular state value function and Adam updates."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.logits = np.zeros((WIDTH * HEIGHT, len(DELTAS)), dtype=float)
        self.values = np.zeros(WIDTH * HEIGHT, dtype=float)
        self.actor_m = np.zeros_like(self.logits)
        self.actor_v = np.zeros_like(self.logits)
        self.critic_m = np.zeros_like(self.values)
        self.critic_v = np.zeros_like(self.values)
        self.updates = 0

    def act(self, state: int, deterministic: bool = False) -> tuple[int, float, float]:
        probs = softmax(self.logits[state])
        action = int(np.argmax(probs)) if deterministic else int(self.rng.choice(4, p=probs))
        return action, float(np.log(probs[action] + 1e-12)), float(self.values[state])

    def update(self, states, actions, rewards, dones, old_logps, old_values,
               next_value: float, expert_lookup: dict[int, int] | None = None,
               fixed_coefficient: float | None = None):
        cfg = self.cfg
        states = np.asarray(states, dtype=int)
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=float)
        old_logps = np.asarray(old_logps, dtype=float)
        old_values = np.asarray(old_values, dtype=float)
        n = len(states)
        advantages = np.zeros(n)
        last_gae = 0.0
        for i in range(n - 1, -1, -1):
            value_after = next_value if i == n - 1 else old_values[i + 1]
            delta = rewards[i] + cfg.gamma * value_after * (1 - dones[i]) - old_values[i]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[i]) * last_gae
            advantages[i] = last_gae
        targets = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        matched_idx = np.array([i for i, s in enumerate(states) if
                                expert_lookup and int(s) in expert_lookup], dtype=int)
        coefficient = (adaptive_coefficient(states, expert_lookup, cfg)
                       if expert_lookup else 0.0)
        if fixed_coefficient is not None and len(matched_idx):
            coefficient = fixed_coefficient
        expert_states = states[matched_idx]
        expert_actions = np.array([expert_lookup[int(s)] for s in expert_states], dtype=int)
        eye = np.eye(4)

        for _ in range(cfg.update_epochs):
            probabilities = softmax(self.logits[states])
            selected = probabilities[np.arange(n), actions]
            ratios = np.exp(np.log(selected + 1e-12) - old_logps)
            active = ((advantages >= 0) & (ratios <= 1 + cfg.clip)) | \
                     ((advantages < 0) & (ratios >= 1 - cfg.clip))
            per_sample = ((active * ratios * advantages)[:, None] *
                          (eye[actions] - probabilities))
            gradient = np.zeros_like(self.logits)
            np.add.at(gradient, states, per_sample / n)

            entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
            entropy_gradient = -probabilities * (np.log(probabilities + 1e-12) +
                                                   entropy[:, None])
            np.add.at(gradient, states, cfg.entropy_coef * entropy_gradient / n)

            if len(expert_states):
                exp_probs = softmax(self.logits[expert_states])
                imitation_gradient = coefficient * (eye[expert_actions] - exp_probs)
                np.add.at(gradient, expert_states,
                          imitation_gradient / len(expert_states))

            value_gradient = np.zeros_like(self.values)
            np.add.at(value_gradient, states, (targets - self.values[states]) / n)
            self.updates += 1
            t = self.updates
            self.actor_m = 0.9 * self.actor_m + 0.1 * gradient
            self.actor_v = 0.999 * self.actor_v + 0.001 * gradient ** 2
            self.critic_m = 0.9 * self.critic_m + 0.1 * value_gradient
            self.critic_v = 0.999 * self.critic_v + 0.001 * value_gradient ** 2
            self.logits += cfg.actor_lr * (self.actor_m / (1 - 0.9 ** t)) / \
                           (np.sqrt(self.actor_v / (1 - 0.999 ** t)) + 1e-8)
            self.values += cfg.critic_lr * (self.critic_m / (1 - 0.9 ** t)) / \
                           (np.sqrt(self.critic_v / (1 - 0.999 ** t)) + 1e-8)
        return coefficient, len(matched_idx)
