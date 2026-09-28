# Goal-selected social imitation for tabular PPO

This repository contains code and saved experiment data for a three-agent grid-world study. A learner uses tabular proximal policy optimization (PPO). It discovers the red goal from reward, selects the observed expert that repeatedly stays at that goal, and adds an action-imitation loss only where learner and expert positions overlap. The comparison uses the same PPO learner without imitation.

## Contents

- `gridworld.py`: six layouts, learner dynamics, rewards, and expert motion.
- `social_ppo.py`: tabular PPO, expert selection, and overlap-weighted imitation.
- `run_experiments.py`: paired baseline and social training and evaluation.
- `run_ablations.py`: fixed-weight and expert-selection ablations.
- `test_experiment.py`: environment and algorithm checks.
- `results/`, `results_maze/`, `results_ablations/`: saved configuration, per-episode data, final-policy evaluations, summary tables, and figures.
- `report/analyze_results.py`, `report/analyze_ablations.py`: analysis of the saved CSV files.

## Reproduce

Python 3.9+ is required. Install packages from `requirements.txt`, then run:

```sh
python3 -m unittest test_experiment.py
python3 run_experiments.py --environment original --episodes 500 --seeds 10 --output results
python3 run_experiments.py --environment maze --episodes 500 --seeds 10 --output results_maze
python3 run_ablations.py --episodes 500 --seeds 10 --output results_ablations
python3 report/analyze_results.py
python3 report/analyze_ablations.py
```

Each method is trained for 500 learner episodes with seeds 0–9 in each layout. Each trained policy is then evaluated for 100 episodes without updates. The learner receives a reward of `-0.1` per step and an additional `+1` on reaching the red goal; episodes stop at the goal or after 100 steps. The learner observes its position, and each expert follows a shortest path to its own goal. The social learner collects expert trajectories during training and uses 10 observed expert episodes for goal-based selection.

The four open layouts and two gated layouts are fixed study environments. The gated layouts contain narrow passages and a nearby expert heading to the wrong goal. The experiments test learning speed under these conditions; they do not establish transfer to unseen maps or noisy experts.

## Saved findings

Social PPO had higher mean return over training episodes 1–100 in all six layouts under ten paired seeds. Both methods generally reached near-shortest routes after 500 episodes. Forcing the wrong-goal expert reduced early return in every layout. A fixed imitation weight of 1.0 performed similarly to the overlap-adaptive weight in these experiments. See the saved `summary.csv` and `REPORT.md` files in each result directory for numerical details.

Authors: Ali Fartoot and Mohammad Amin Ghanizadeh, equal contributors; Department of Electrical and Computer Engineering, College of Engineering, University of Tehran, Tehran, Iran.
