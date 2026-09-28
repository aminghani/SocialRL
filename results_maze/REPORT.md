# Obstacle environment experiment

This is a new experiment using the two gated layouts in
[scenario_maps.png](scenario_maps.png). It uses 10 paired seeds, 500 learner
training episodes per method and layout, and the same reward, 100-step limit,
PPO settings, and evaluation protocol as the original experiment. The walls
force a 19-step shortest route from the learner to the red goal, with two
separate passages. The blue expert begins near the learner but ends at the
wrong goal; the red expert demonstrates the route through both passages.
The exact blocked cells and settings are in [config.json](config.json).

| Layout | First 100 episodes: baseline return | First 100 episodes: social return | Social improvement | First 100 success: baseline / social | Median episode reaching 90% success: baseline / social |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lower gate | −6.177 | −4.712 | +1.465 | 53.7% / 62.2% | 77.5 / 59.0 |
| Upper gate | −6.396 | −4.421 | +1.976 | 49.5% / 65.1% | 77.0 / 55.0 |

The paired-seed 95% t intervals for the first-100-episode return improvement
were **+0.755 to +2.175** in the lower-gate layout and **+0.790 to +3.162**
in the upper-gate layout. The 90% success milestone uses a trailing 25-episode
window, computed separately per seed. In all 20 social trials, the goal-based
vote selected the red expert by 10–0. Among social training episodes 11–100,
the mean adaptive imitation coefficient was about 0.61 and 0.60 for the two
layouts, respectively; expert actions matched at least one learner state in
about 76% and 78% of those episodes.

At the end of training, the methods were close. Final sampled-policy mean
returns were −0.948 baseline versus −0.902 social in the lower-gate layout,
and −0.931 baseline versus −0.934 social in the upper-gate layout. Final goal
success was 100% for both methods in the lower-gate layout; in the upper-gate
layout it was 100% baseline and 99.9% social. The evidence here supports a
**sample-efficiency benefit**, not a claim of better asymptotic success.

The [return plot](return_comparison.png) and
[success plot](success_comparison.png) show 25-episode moving averages; shaded
bands are normal-approximation 95% confidence intervals across seeds. The
[episode metrics](episode_metrics.csv),
[per-seed evaluations](trial_evaluations.csv), and
[aggregate summary](summary.csv) provide the underlying numbers. The new
maps and conclusions depend on the explicit environment assumptions in
[../README.md](../README.md).
