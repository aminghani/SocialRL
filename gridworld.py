"""Three-agent grid worlds with open layouts and gated mazes.

Coordinates are (x, y), zero based, inside the 12 by 12 free interior.
The one-cell outer border is represented by boundary checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


WIDTH = 12
HEIGHT = 12
UP, RIGHT, DOWN, LEFT = range(4)
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class Scenario:
    name: str
    learner_start: tuple[int, int]
    red_expert_start: tuple[int, int]
    blue_expert_start: tuple[int, int]
    red_goal: tuple[int, int]
    blue_goal: tuple[int, int]
    blocked: frozenset[tuple[int, int]] = frozenset()


SCENARIOS = (
    Scenario("scenario_1", (0, 11), (0, 1), (0, 0), (10, 8), (10, 4)),
    Scenario("scenario_2", (0, 11), (0, 9), (0, 0), (10, 8), (10, 4)),
    Scenario("scenario_3", (9, 10), (0, 1), (0, 0), (10, 10), (10, 4)),
    Scenario("scenario_4", (0, 10), (0, 1), (0, 0), (10, 10), (10, 4)),
)


MAZE_SCENARIOS = (
    Scenario(
        "maze_lower_gate", (0, 10), (0, 9), (1, 10), (10, 1), (10, 10),
        frozenset({(5, y) for y in range(HEIGHT) if y not in (8, 9)} |
                  {(x, 4) for x in range(6, WIDTH) if x not in (9, 10)}),
    ),
    Scenario(
        "maze_upper_gate", (0, 1), (0, 2), (1, 1), (10, 10), (10, 1),
        frozenset({(5, y) for y in range(HEIGHT) if y not in (2, 3)} |
                  {(x, 7) for x in range(6, WIDTH) if x not in (8, 9)}),
    ),
)


def move(position: tuple[int, int], action: int,
         blocked: frozenset[tuple[int, int]] = frozenset()) -> tuple[int, int]:
    dx, dy = DELTAS[action]
    x = min(WIDTH - 1, max(0, position[0] + dx))
    y = min(HEIGHT - 1, max(0, position[1] + dy))
    return position if (x, y) in blocked else (x, y)


def distance_map(goal: tuple[int, int], blocked: frozenset[tuple[int, int]]):
    """Shortest navigable distance from each cell to a goal."""
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        position = queue.popleft()
        for action in range(4):
            neighbor = move(position, action, blocked)
            if neighbor not in distances:
                distances[neighbor] = distances[position] + 1
                queue.append(neighbor)
    return distances


def state_id(position: tuple[int, int]) -> int:
    return position[1] * WIDTH + position[0]


class GridWorld:
    """Three independent agents; a learner episode ends at red or at the limit."""

    def __init__(self, scenario: Scenario, max_steps: int = 100,
                 rng: np.random.Generator | None = None):
        self.scenario = scenario
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.expert_distances = [distance_map(scenario.red_goal, scenario.blocked),
                                 distance_map(scenario.blue_goal, scenario.blocked)]
        for position in (scenario.learner_start, scenario.red_expert_start,
                         scenario.blue_expert_start, scenario.red_goal, scenario.blue_goal):
            if position in scenario.blocked:
                raise ValueError(f"An agent or goal is inside a wall: {position}")
        if scenario.learner_start not in self.expert_distances[0]:
            raise ValueError("Learner cannot reach the red goal")
        if (scenario.red_expert_start not in self.expert_distances[0] or
                scenario.blue_expert_start not in self.expert_distances[1]):
            raise ValueError("An expert cannot reach its goal")
        self.reset()

    def reset(self) -> int:
        self.position = self.scenario.learner_start
        self.steps = 0
        self.expert_positions = [self.scenario.red_expert_start,
                                 self.scenario.blue_expert_start]
        self.expert_goals = [self.scenario.red_goal, self.scenario.blue_goal]
        self.expert_histories = [[self.expert_positions[0]], [self.expert_positions[1]]]
        self.expert_actions = [[], []]
        return state_id(self.position)

    def _advance_experts(self):
        for index, (position, goal) in enumerate(zip(self.expert_positions, self.expert_goals)):
            if position == goal:
                self.expert_histories[index].append(position)
                continue
            distance = self.expert_distances[index][position]
            candidates = [action for action in range(4)
                          if self.expert_distances[index].get(
                              move(position, action, self.scenario.blocked), float("inf")
                          ) < distance]
            action = int(self.rng.choice(candidates))
            self.expert_actions[index].append(action)
            self.expert_positions[index] = move(position, action, self.scenario.blocked)
            self.expert_histories[index].append(self.expert_positions[index])

    def complete_experts(self, stationary_window: int):
        """Let experts finish if the learner episode ended before they did."""
        while any(p != g for p, g in zip(self.expert_positions, self.expert_goals)):
            self._advance_experts()
        for _ in range(stationary_window):
            self._advance_experts()
        return [(states.copy(), actions.copy()) for states, actions in
                zip(self.expert_histories, self.expert_actions)]

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        self.position = move(self.position, action, self.scenario.blocked)
        self._advance_experts()
        self.steps += 1
        reached_goal = self.position == self.scenario.red_goal
        truncated = self.steps >= self.max_steps and not reached_goal
        # A step costs 0.1 and reaching the red goal pays an additional 1.
        reward = -0.1 + (1.0 if reached_goal else 0.0)
        return state_id(self.position), reward, reached_goal, truncated
