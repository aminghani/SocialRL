"""Regenerate paired ablation comparisons from the three saved result folders."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
T_CRIT_95_DF9 = 2.262157
METHODS = ("baseline_ppo", "social_ppo", "fixed_025", "fixed_100",
           "balanced_expert", "wrong_expert")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def interval(differences):
    values = np.asarray(differences, dtype=float)
    center = float(np.mean(values))
    half = float(T_CRIT_95_DF9 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return [center - half, center + half]


def main():
    rows, evaluations = [], []
    for directory in ("results", "results_maze", "results_ablations"):
        rows += read_csv(ROOT / directory / "episode_metrics.csv")
        evaluations += read_csv(ROOT / directory / "trial_evaluations.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"], row["method"], int(row["seed"])].append(row)
    evals = {(row["scenario"], row["method"], int(row["seed"])): row
             for row in evaluations}
    scenes = ["scenario_1", "scenario_2", "scenario_3", "scenario_4",
              "maze_lower_gate", "maze_upper_gate"]
    output = {}
    for scene in scenes:
        output[scene] = {}
        full_early = np.array([np.mean([
            float(row["return"]) for row in grouped[scene, "social_ppo", seed][:100]
        ]) for seed in range(10)])
        full_final = np.array([float(evals[scene, "social_ppo", seed]["evaluation_mean_return"])
                               for seed in range(10)])
        for method in METHODS:
            early = np.array([np.mean([
                float(row["return"]) for row in grouped[scene, method, seed][:100]
            ]) for seed in range(10)])
            final = np.array([float(evals[scene, method, seed]["evaluation_mean_return"])
                              for seed in range(10)])
            successes = np.array([np.mean([
                int(row["success"]) for row in grouped[scene, method, seed][:100]
            ]) for seed in range(10)])
            output[scene][method] = {
                "first100_return": float(early.mean()),
                "first100_success": float(successes.mean()),
                "final_return": float(final.mean()),
                "early_minus_full_ci": interval(early - full_early),
                "final_minus_full_ci": interval(final - full_final),
            }
        output[scene]["balanced_subgroups"] = {
            label: {
                "first100_return": float(np.mean([np.mean([
                    float(row["return"]) for row in
                    grouped[scene, "balanced_expert", seed][:100]
                ]) for seed in seeds])),
                "selected_expert": expert,
            }
            for label, seeds, expert in (
                ("red_even_seeds", range(0, 10, 2), 0),
                ("blue_odd_seeds", range(1, 10, 2), 1),
            )
        }
    target = Path(__file__).resolve().parent / "ablation_stats.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
