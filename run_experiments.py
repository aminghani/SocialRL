"""Run paired baseline/social PPO experiments and write data and plots.

Usage: python3 run_experiments.py --environment maze --seeds 10 --episodes 500
"""

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

from gridworld import GridWorld, HEIGHT, MAZE_SCENARIOS, SCENARIOS, WIDTH, Scenario
from social_ppo import Config, TabularPPO, select_expert_and_lookup


def train(scenario: Scenario, method: str, seed: int, cfg: Config):
    allowed = {"baseline_ppo", "social_ppo", "fixed_025", "fixed_100",
               "balanced_expert", "wrong_expert"}
    if method not in allowed:
        raise ValueError(f"Unknown method: {method}")
    social_variant = method != "baseline_ppo"
    fixed_coefficient = {"fixed_025": 0.25, "fixed_100": 1.0}.get(method)
    forced_expert = (seed % 2 if method == "balanced_expert" else
                     1 if method == "wrong_expert" else None)
    rng = np.random.default_rng(seed)
    demo_rng = np.random.default_rng(seed + 100_000)
    agent = TabularPPO(cfg, rng)
    env = GridWorld(scenario, cfg.max_steps, demo_rng)
    lookup = {}
    selected_expert = None
    votes = None
    discovery_episode = None
    discovered_goal = None
    expert_traces = [[], []]
    rows = []

    for episode in range(1, cfg.episodes + 1):
        state = env.reset()
        states, actions, rewards, dones, logps, values = [], [], [], [], [], []
        reached_goal = False
        for _ in range(cfg.max_steps):
            action, logp, value = agent.act(state)
            next_state, reward, terminal, truncated = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminal)
            logps.append(logp)
            values.append(value)
            state = next_state
            if terminal or truncated:
                reached_goal = terminal
                break

        next_value = 0.0 if reached_goal else float(agent.values[state])
        if social_variant:
            for expert_index, trace in enumerate(env.complete_experts(cfg.stationary_window)):
                expert_traces[expert_index].append(trace)
        coefficient, matches = agent.update(
            states, actions, rewards, dones, logps, values, next_value,
            lookup if social_variant else None, fixed_coefficient,
        )
        if social_variant and reached_goal and discovery_episode is None:
            discovery_episode = episode
            discovered_goal = env.position
        if (social_variant and discovery_episode is not None
                and selected_expert is None
                and len(expert_traces[0]) >= cfg.demo_episodes):
            selected_expert, lookup, votes = select_expert_and_lookup(
                scenario, discovered_goal, demo_rng, cfg,
                [traces[-cfg.demo_episodes:] for traces in expert_traces],
                forced_expert,
            )
        rows.append({
            "scenario": scenario.name, "method": method, "seed": seed,
            "episode": episode, "return": float(sum(rewards)),
            "success": int(reached_goal), "steps": len(rewards),
            "coefficient": coefficient, "expert_matches": matches,
            "goal_discovered": int(discovery_episode is not None),
        })

    evaluation = evaluate(agent, scenario, seed + 200_000, cfg)
    evaluation.update({
        "scenario": scenario.name, "method": method, "seed": seed,
        "discovery_episode": discovery_episode,
        "selected_expert": selected_expert,
        "expert_votes": votes,
        "lookup_states": len(lookup),
    })
    return rows, evaluation


def evaluate(agent: TabularPPO, scenario: Scenario, seed: int, cfg: Config,
             episodes: int = 100):
    # The policy is sampled, as in training, but no learning occurs.
    agent.rng = np.random.default_rng(seed)
    env = GridWorld(scenario, cfg.max_steps)
    rewards, successes, steps = [], [], []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for step in range(1, cfg.max_steps + 1):
            action, _, _ = agent.act(state)
            state, reward, terminal, truncated = env.step(action)
            total += reward
            if terminal or truncated:
                break
        rewards.append(total)
        successes.append(int(terminal))
        steps.append(step)
    return {
        "evaluation_episodes": episodes,
        "evaluation_mean_return": float(np.mean(rewards)),
        "evaluation_success_rate": float(np.mean(successes)),
        "evaluation_mean_steps": float(np.mean(steps)),
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    indices = np.arange(1, len(values) + 1)
    starts = np.maximum(0, indices - window)
    return (cumulative[indices] - cumulative[starts]) / (indices - starts)


def mean_ci(values: np.ndarray):
    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        return mean, np.zeros_like(mean)
    # Normal approximation, explicitly labeled in figures and report.
    half_width = 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, half_width


def plot_scenario_map(ax, scenario: Scenario):
    ax.set_facecolor("white")
    for x in range(WIDTH + 1):
        ax.axvline(x - .5, lw=.3, color="#999999")
    for y in range(HEIGHT + 1):
        ax.axhline(y - .5, lw=.3, color="#999999")
    for x, y in scenario.blocked:
        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color="#454954"))
    positions = [
        (scenario.learner_start, "L", "#2d8a4d"),
        (scenario.red_expert_start, "E₁", "#6bbb75"),
        (scenario.blue_expert_start, "E₂", "#46d353"),
        (scenario.red_goal, "R", "#e53935"),
        (scenario.blue_goal, "B", "#244bdb"),
    ]
    for (x, y), label, color in positions:
        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, color=color))
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
    ax.set_xlim(-.5, WIDTH - .5)
    ax.set_ylim(HEIGHT - .5, -.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(scenario.name.replace("_", " ").title())


def plot_comparisons(rows: list[dict], output: Path, cfg: Config,
                     scenarios: tuple[Scenario, ...],
                     seeds: list[int], window: int):
    methods = ("baseline_ppo", "social_ppo")
    colors = {"baseline_ppo": "#e68613", "social_ppo": "#2858cb"}
    labels = {"baseline_ppo": "Baseline PPO", "social_ppo": "Social PPO"}
    by_key = {(r["scenario"], r["method"], r["seed"], r["episode"]): r
              for r in rows}
    nrows, ncols = (2, 2) if len(scenarios) == 4 else (1, len(scenarios))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, scenario in zip(axes.flat, scenarios):
        for method in methods:
            curves = np.array([
                rolling_mean(np.array([
                    by_key[scenario.name, method, seed, ep]["return"]
                    for ep in range(1, cfg.episodes + 1)]), window)
                for seed in seeds
            ])
            mean, ci = mean_ci(curves)
            x = np.arange(1, cfg.episodes + 1)
            ax.plot(x, mean, label=labels[method], color=colors[method], lw=2)
            ax.fill_between(x, mean - ci, mean + ci, color=colors[method], alpha=.17)
        ax.set_title(scenario.name.replace("_", " ").title())
        ax.grid(alpha=.25)
        ax.set_ylim(-0.1 * cfg.max_steps - .5, 1.2)
    for ax in axes[-1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Mean return ({window}-episode moving average)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"PPO comparison across {len(seeds)} paired seeds; bands: approximate 95% CI")
    fig.tight_layout()
    fig.savefig(output / "return_comparison.png", dpi=180)
    fig.savefig(output / "return_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, scenario in zip(axes.flat, scenarios):
        for method in methods:
            curves = np.array([
                rolling_mean(np.array([
                    by_key[scenario.name, method, seed, ep]["success"]
                    for ep in range(1, cfg.episodes + 1)]), window)
                for seed in seeds
            ])
            mean, ci = mean_ci(curves)
            x = np.arange(1, cfg.episodes + 1)
            ax.plot(x, mean, label=labels[method], color=colors[method], lw=2)
            ax.fill_between(x, np.maximum(0, mean - ci), np.minimum(1, mean + ci),
                            color=colors[method], alpha=.17)
        ax.set_title(scenario.name.replace("_", " ").title())
        ax.grid(alpha=.25)
        ax.set_ylim(0, 1.05)
    for ax in axes[-1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Success rate ({window}-episode moving average)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"Goal reaching across {len(seeds)} paired seeds; bands: approximate 95% CI")
    fig.tight_layout()
    fig.savefig(output / "success_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    for ax, scenario in zip(np.atleast_1d(axes).flat, scenarios):
        plot_scenario_map(ax, scenario)
    fig.tight_layout()
    fig.savefig(output / "scenario_maps.png", dpi=180)
    plt.close(fig)


def summarize(rows: list[dict], evaluations: list[dict], cfg: Config,
              scenarios: tuple[Scenario, ...], seeds: list[int]):
    result = []
    for scenario in scenarios:
        for method in ("baseline_ppo", "social_ppo"):
            trial_evals = [r for r in evaluations if r["scenario"] == scenario.name
                           and r["method"] == method]
            trial_rows = [r for r in rows if r["scenario"] == scenario.name
                          and r["method"] == method]
            final = [r for r in trial_rows if r["episode"] > cfg.episodes - 50]
            result.append({
                "scenario": scenario.name, "method": method, "seeds": len(seeds),
                "last_50_training_return": float(np.mean([r["return"] for r in final])),
                "last_50_training_success": float(np.mean([r["success"] for r in final])),
                "evaluation_return": float(np.mean([r["evaluation_mean_return"] for r in trial_evals])),
                "evaluation_success": float(np.mean([r["evaluation_success_rate"] for r in trial_evals])),
                "evaluation_steps": float(np.mean([r["evaluation_mean_steps"] for r in trial_evals])),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--environment", choices=("original", "maze"), default="original")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 50 or args.seeds < 1 or args.window < 1:
        parser.error("episodes must be >= 50; seeds and window must be >= 1")
    scenarios = SCENARIOS if args.environment == "original" else MAZE_SCENARIOS
    if args.output is None:
        args.output = Path("results" if args.environment == "original" else "results_maze")
    cfg = Config(episodes=args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    rows, evaluations = [], []
    for scenario in scenarios:
        for seed in seeds:
            for method in ("baseline_ppo", "social_ppo"):
                trial_rows, evaluation = train(scenario, method, seed, cfg)
                rows.extend(trial_rows)
                evaluations.append(evaluation)
        print(f"Completed {scenario.name}: {len(seeds)} paired seeds", flush=True)

    summary = summarize(rows, evaluations, cfg, scenarios, seeds)
    write_csv(args.output / "episode_metrics.csv", rows)
    write_csv(args.output / "trial_evaluations.csv", evaluations)
    write_csv(args.output / "summary.csv", summary)
    with (args.output / "config.json").open("w") as handle:
        json.dump({"environment": args.environment, "config": asdict(cfg), "seeds": seeds,
                   "moving_average_window": args.window,
                   "scenarios": [{**asdict(s), "blocked": sorted(map(list, s.blocked))}
                                 for s in scenarios]}, handle, indent=2)
    plot_comparisons(rows, args.output, cfg, scenarios, seeds, args.window)
    for item in summary:
        print(f"{item['scenario']:>10} {item['method']:>12}: "
              f"eval return={item['evaluation_return']:.2f}, "
              f"success={item['evaluation_success']:.1%}")


if __name__ == "__main__":
    main()
