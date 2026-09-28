import unittest

import numpy as np

from gridworld import GridWorld, MAZE_SCENARIOS, SCENARIOS, RIGHT, WIDTH, move, state_id
from social_ppo import Config, TabularPPO, adaptive_coefficient, select_expert_and_lookup


class ExperimentTests(unittest.TestCase):
    def test_wall_goal_and_three_agent_trajectories(self):
        scenario = SCENARIOS[2]
        env = GridWorld(scenario, rng=np.random.default_rng(42))
        self.assertEqual(len(env.expert_positions), 2)
        self.assertEqual(env.reset(), state_id(scenario.learner_start))
        state, reward, terminal, truncated = env.step(RIGHT)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 0.9)
        self.assertEqual(state, state_id(scenario.red_goal))
        traces = env.complete_experts(3)
        self.assertEqual(traces[0][0][-1], scenario.red_goal)
        self.assertEqual(traces[1][0][-1], scenario.blue_goal)
        self.assertEqual(len(traces[0][0]), len(traces[0][1]) + 4)

    def test_expert_vote_uses_stationary_matching_goal(self):
        cfg = Config(demo_episodes=10)
        env = GridWorld(SCENARIOS[0], rng=np.random.default_rng(9))
        traces = [[], []]
        for _ in range(10):
            env.reset()
            completed = env.complete_experts(cfg.stationary_window)
            for index in (0, 1):
                traces[index].append(completed[index])
        red_choice, red_lookup, red_votes = select_expert_and_lookup(
            SCENARIOS[0], SCENARIOS[0].red_goal, np.random.default_rng(1), cfg, traces
        )
        blue_choice, blue_lookup, blue_votes = select_expert_and_lookup(
            SCENARIOS[0], SCENARIOS[0].blue_goal, np.random.default_rng(1), cfg, traces
        )
        self.assertEqual((red_choice, red_votes), (0, [10, 0]))
        self.assertEqual((blue_choice, blue_votes), (1, [0, 10]))
        self.assertTrue(red_lookup and blue_lookup)
        forced_choice, _, forced_votes = select_expert_and_lookup(
            SCENARIOS[0], SCENARIOS[0].red_goal, np.random.default_rng(1), cfg,
            traces, forced_expert=1
        )
        self.assertEqual((forced_choice, forced_votes), (1, [10, 0]))

    def test_jaccard_weight_and_decay(self):
        cfg = Config()
        lookup = {0: RIGHT, 1: RIGHT}
        self.assertAlmostEqual(adaptive_coefficient(np.array([0, 1, 2, 3]), lookup, cfg), .5)
        self.assertAlmostEqual(adaptive_coefficient(np.array([0] + list(range(2, 21))),
                                                      lookup, cfg),
                               .05 * np.exp(-5 * (.1 - .05)))
        self.assertEqual(adaptive_coefficient(np.array([WIDTH + 1]), lookup, cfg), 0)

    def test_maze_walls_and_expert_paths(self):
        for scenario in MAZE_SCENARIOS:
            env = GridWorld(scenario, rng=np.random.default_rng(13))
            self.assertEqual(env.expert_distances[0][scenario.learner_start], 19)
            self.assertEqual(env.expert_distances[0][scenario.red_expert_start], 18)
            # A move into any obstacle leaves the agent in its prior cell.
            self.assertTrue(any(
                move((x - 1, y), RIGHT, scenario.blocked) == (x - 1, y)
                for x, y in scenario.blocked if x > 0 and (x - 1, y) not in scenario.blocked
            ))
            traces = env.complete_experts(3)
            for index, (_, goal) in enumerate(((scenario.red_expert_start, scenario.red_goal),
                                               (scenario.blue_expert_start, scenario.blue_goal))):
                states, actions = traces[index]
                self.assertEqual(states[-1], goal)
                self.assertFalse(any(position in scenario.blocked for position in states))
                self.assertEqual(len(actions), env.expert_distances[index][states[0]])

    def test_fixed_imitation_weight(self):
        cfg = Config(update_epochs=1)
        learner = TabularPPO(cfg, np.random.default_rng(2))
        coefficient, matches = learner.update(
            states=[0, 1], actions=[RIGHT, RIGHT], rewards=[-.1, -.1],
            dones=[False, False], old_logps=[np.log(.25), np.log(.25)],
            old_values=[0., 0.], next_value=0., expert_lookup={0: RIGHT},
            fixed_coefficient=.25,
        )
        self.assertEqual(matches, 1)
        self.assertAlmostEqual(coefficient, .25)


if __name__ == "__main__":
    unittest.main()
