# Experiment results

Run: 4 scenarios × 2 methods × 10 paired seeds × 500 training episodes.
Each final policy was evaluated in 100 additional sampled episodes. The complete
settings are in [config.json](config.json), and the implementation choices are
documented in [../README.md](../README.md).

| Scenario | First 100 episodes: baseline return | First 100 episodes: social return | Social minus baseline | Final evaluation return: baseline / social |
| --- | ---: | ---: | ---: | ---: |
| 1 | −3.373 | −2.270 | +1.103 | −0.349 / −0.322 |
| 2 | −3.373 | −2.307 | +1.067 | −0.349 / −0.302 |
| 3 | +0.539 | +0.610 | +0.072 | +0.899 / +0.900 |
| 4 | −3.640 | −1.953 | +1.687 | −0.248 / −0.103 |

Social PPO had higher average return over the first 100 training episodes in
all four scenarios. The paired-seed 95% t intervals for that improvement were
approximately +0.802 to +1.404, +0.723 to +1.410, +0.025 to +0.118, and
+1.167 to +2.206, respectively. At the end, both methods reached the red
goal in essentially every evaluation episode (100% in the aggregate for each
scenario and method). The main difference in these runs was learning speed and,
in scenario 4, the length of the learned route. All 40 social trials selected
the red-goal expert, each with a 10–0 vote against the blue-goal expert.

The baseline curves for scenarios 1 and 2 are identical because the learner's
start and red goal are identical, and the independent experts do not change
the learner's transitions. The social curves differ because the red expert
starts in a different cell and therefore supplies different demonstrations.

These experiments use the explicit reward, movement, observation, and tabular
PPO choices documented in the README.

## Files

- [return_comparison.png](return_comparison.png) and [return_comparison.pdf](return_comparison.pdf): training return curves.
- [success_comparison.png](success_comparison.png): goal-reaching curves.
- [scenario_maps.png](scenario_maps.png): the four transcribed layouts.
- [episode_metrics.csv](episode_metrics.csv): every learner episode.
- [trial_evaluations.csv](trial_evaluations.csv): final policy evaluation per seed.
- [summary.csv](summary.csv): aggregate late-training and evaluation metrics.
