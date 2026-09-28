"""Run fixed-coefficient and expert-selection ablations on all six layouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rl_paper_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import MAZE_SCENARIOS, SCENARIOS
from run_experiments import rolling_mean, train, write_csv
from social_ppo import Config


VARIANTS = ("fixed_025", "fixed_100", "balanced_expert", "wrong_expert")
COLORS = {
    "baseline_ppo": "#dc8310", "social_ppo": "#2355c7",
    "fixed_025": "#954ea4", "fixed_100": "#c84040",
    "balanced_expert": "#258d66", "wrong_expert": "#555b66",
}
LABELS = {
    "baseline_ppo": "PPO baseline", "social_ppo": "Adaptive social",
    "fixed_025": "Fixed λ=0.25", "fixed_100": "Fixed λ=1.0",
    "balanced_expert": "No vote (5/5)", "wrong_expert": "Wrong expert",
}


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matching_controls(cfg: Config, seeds: list[int]):
    rows, evaluations = [], []
    for directory in (Path("results"), Path("results_maze")):
        config = json.loads((directory / "config.json").read_text())
        if config["config"] != asdict(cfg) or config["seeds"] != seeds:
            return None, None
        rows.extend(read_csv(directory / "episode_metrics.csv"))
        evaluations.extend(read_csv(directory / "trial_evaluations.csv"))
    return rows, evaluations


def summarize(rows, evaluations, scenarios, methods, cfg: Config):
    output = []
    for scenario in scenarios:
        for method in methods:
            subset = [r for r in rows if r["scenario"] == scenario.name
                      and r["method"] == method]
            early = [r for r in subset if int(r["episode"]) <= 100]
            late = [r for r in subset if int(r["episode"]) > cfg.episodes - 50]
            trials = [r for r in evaluations if r["scenario"] == scenario.name
                      and r["method"] == method]
            output.append({
                "scenario": scenario.name, "method": method,
                "first_100_return": float(np.mean([float(r["return"]) for r in early])),
                "first_100_success": float(np.mean([int(r["success"]) for r in early])),
                "last_50_return": float(np.mean([float(r["return"]) for r in late])),
                "evaluation_return": float(np.mean([
                    float(r["evaluation_mean_return"]) for r in trials])),
                "evaluation_success": float(np.mean([
                    float(r["evaluation_success_rate"]) for r in trials])),
            })
    return output


def plot_factor(rows, scenarios, seeds, methods, output: Path, filename: str,
                title: str, window: int = 25, horizon: int = 200):
    keyed = {(r["scenario"], r["method"], int(r["seed"]), int(r["episode"])):
             float(r["return"]) for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, scenarios):
        for method in methods:
            curves = np.array([
                rolling_mean(np.array([
                    keyed[scenario.name, method, seed, episode]
                    for episode in range(1, horizon + 1)]), window)
                for seed in seeds
            ])
            ax.plot(np.arange(1, horizon + 1), np.mean(curves, axis=0),
                    label=LABELS[method], color=COLORS[method], lw=1.8)
        ax.set_title(scenario.name.replace("_", " ").title())
        ax.set_ylim(-10.5, 1.3)
        ax.grid(alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Return ({window}-episode moving average)")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results_ablations"))
    args = parser.parse_args()
    if args.episodes < 100 or args.seeds < 1:
        parser.error("episodes must be >=100 and seeds must be >=1")
    cfg = Config(episodes=args.episodes)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    scenarios = SCENARIOS + MAZE_SCENARIOS
    args.output.mkdir(parents=True, exist_ok=True)
    rows, evaluations = [], []
    for scenario in scenarios:
        for seed in seeds:
            for method in VARIANTS:
                trial_rows, evaluation = train(scenario, method, seed, cfg)
                rows.extend(trial_rows)
                evaluations.append(evaluation)
        print(f"Completed {scenario.name}: {len(seeds)} seeds × {len(VARIANTS)} ablations",
              flush=True)
    write_csv(args.output / "episode_metrics.csv", rows)
    write_csv(args.output / "trial_evaluations.csv", evaluations)
    (args.output / "config.json").write_text(json.dumps({
        "config": asdict(cfg), "seeds": seeds,
        "variants": {
            "fixed_025": "Goal-voted red expert, fixed imitation coefficient 0.25",
            "fixed_100": "Goal-voted red expert, fixed imitation coefficient 1.0",
            "balanced_expert": "No goal vote: red on even seeds, blue on odd seeds",
            "wrong_expert": "Forced blue expert in every seed, adaptive coefficient",
        },
        "control_sources": ["results", "results_maze"],
    }, indent=2) + "\n")

    control_rows, control_evals = matching_controls(cfg, seeds)
    if control_rows is not None:
        all_rows = control_rows + rows
        all_evals = control_evals + evaluations
        methods = ("baseline_ppo", "social_ppo") + VARIANTS
        write_csv(args.output / "summary.csv",
                  summarize(all_rows, all_evals, scenarios, methods, cfg))
        plot_factor(all_rows, scenarios, seeds,
                    ("baseline_ppo", "social_ppo", "fixed_025", "fixed_100"),
                    args.output, "coefficient_ablation.png",
                    "Imitation coefficient ablation: first 200 episodes")
        plot_factor(all_rows, scenarios, seeds,
                    ("baseline_ppo", "social_ppo", "balanced_expert", "wrong_expert"),
                    args.output, "selection_ablation.png",
                    "Expert selection ablation: first 200 episodes")
        print("Wrote combined summaries and plots using matching saved controls")
    else:
        write_csv(args.output / "summary.csv",
                  summarize(rows, evaluations, scenarios, VARIANTS, cfg))
        print("Saved controls have different settings; comparison plots skipped")


if __name__ == "__main__":
    main()
