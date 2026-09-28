# Ablation experiments

Four variants were trained on each of the four original and two gated layouts:
10 seeds and 500 learner episodes per variant and layout. The stored baseline
PPO and adaptive Social PPO runs in `results/` and `results_maze/` serve as
paired controls. All other settings, including goal discovery, expert
trajectories, and the training budget, were held fixed.

| Variant | Change from adaptive Social PPO | Question |
| --- | --- | --- |
| `fixed_025` | Use a constant imitation coefficient of 0.25 when demonstrations match | Is the adaptive value better than a modest fixed weight? |
| `fixed_100` | Use a constant imitation coefficient of 1.0 when demonstrations match | Is adaptivity needed at a strong fixed weight? |
| `balanced_expert` | Omit the goal vote; select red on even seeds and blue on odd seeds | What happens when expert identity is chosen without goal evidence? |
| `wrong_expert` | Force the blue-goal expert in every seed | How much can an incorrect demonstration hurt? |

The balanced assignment is deterministic and has exactly five correct and
five incorrect selections per layout. The wrong-expert variant is a stress
test, rather than a model of unbiased random choice.

## Main results

Mean training return over episodes 1–100 (higher is better):

| Layout | PPO baseline | Adaptive Social | Fixed 0.25 | Fixed 1.0 | Balanced expert | Wrong expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scenario 1 | −3.373 | −2.270 | −2.391 | −2.193 | −2.640 | −3.026 |
| Scenario 2 | −3.373 | −2.307 | −2.407 | −2.192 | −2.796 | −3.151 |
| Scenario 3 | +0.539 | +0.610 | +0.594 | +0.610 | +0.581 | +0.539 |
| Scenario 4 | −3.640 | −1.953 | −2.271 | −1.968 | −2.615 | −3.370 |
| Lower gate | −6.177 | −4.712 | −4.817 | −4.614 | −5.455 | −5.954 |
| Upper gate | −6.396 | −4.421 | −4.676 | −4.329 | −5.959 | −7.042 |

The goal-vote component has strong evidence in these tests. Forcing the wrong
expert lowered first-100-episode return relative to adaptive Social PPO in
every layout; the paired-seed 95% intervals for the decrease excluded zero in
all six. In the upper-gate maze, it even fell below the plain PPO baseline
(−7.042 versus −6.396). The balanced no-vote variant sat between the fully
selected and wrong-expert variants in most layouts.

The coefficient ablation gives a narrower conclusion. Fixed λ=1.0 matched or
slightly exceeded adaptive Social PPO in early mean return in several layouts;
the paired 95% interval for its difference from adaptive included zero in all
six. Fixed λ=0.25 was worse than adaptive in Scenario 3, Scenario 4, and the
upper-gate maze under the same paired-interval analysis. These runs **do not
establish that the Jaccard adaptation is necessary** in these fixed layouts.
They do show that choosing the appropriate expert matters.

Final evaluation return in the gated mazes remained lower with a forced wrong
expert: −1.254 versus −0.902 in the lower gate and −1.458 versus −0.934 in the
upper gate. Paired 95% intervals for wrong minus adaptive were
[−0.470, −0.234] and [−0.686, −0.363], respectively. A fixed λ=1.0 did
not show a clear final disadvantage against adaptive in either maze.

## Files

- [coefficient_ablation.png](coefficient_ablation.png): early return curves for adaptive and fixed weights.
- [selection_ablation.png](selection_ablation.png): early return curves for goal-voted, balanced, and wrong-expert selection.
- [episode_metrics.csv](episode_metrics.csv): every ablation training episode.
- [trial_evaluations.csv](trial_evaluations.csv): final evaluation for each ablation seed.
- [summary.csv](summary.csv): all methods, including the saved controls.
- [config.json](config.json): variant definitions and experiment settings.

Paired intervals and subgroup checks can be regenerated using
`python3 report/analyze_ablations.py` from the project root. The resulting
`report/ablation_stats.json` supplies the extended technical report.
