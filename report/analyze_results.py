"""Regenerate the numerical tables used in Complete_Report.tex from saved CSVs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TRIALS = 10
T_CRIT_95_DF9 = 2.262157


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_interval(differences):
    differences = np.asarray(differences, dtype=float)
    mean = float(np.mean(differences))
    half = float(T_CRIT_95_DF9 * np.std(differences, ddof=1) / np.sqrt(len(differences)))
    return [mean - half, mean + half]


def milestone(successes, window=25, threshold=.9):
    values = np.asarray(successes, dtype=float)
    for end in range(window, len(values) + 1):
        if np.mean(values[end - window:end]) >= threshold:
            return end
    return None


def analyze_folder(folder):
    episodes = read_csv(folder / "episode_metrics.csv")
    evaluations = read_csv(folder / "trial_evaluations.csv")
    config = json.loads((folder / "config.json").read_text())
    by_trial = defaultdict(list)
    for row in episodes:
        by_trial[row["scenario"], row["method"], int(row["seed"])].append(row)
    eval_by_trial = {(row["scenario"], row["method"], int(row["seed"])): row
                     for row in evaluations}
    output = {}
    seeds = config["seeds"]
    assert len(seeds) == TRIALS
    for scene in config["scenarios"]:
        name = scene["name"]
        early = {}
        final = {}
        success_early = {}
        milestones = {}
        for method in ("baseline_ppo", "social_ppo"):
            early[method] = [float(np.mean([
                float(row["return"]) for row in by_trial[name, method, seed][:100]
            ])) for seed in seeds]
            final[method] = [float(eval_by_trial[name, method, seed]["evaluation_mean_return"])
                             for seed in seeds]
            success_early[method] = [float(np.mean([
                int(row["success"]) for row in by_trial[name, method, seed][:100]
            ])) for seed in seeds]
            milestones[method] = [milestone([
                int(row["success"]) for row in by_trial[name, method, seed]
            ]) for seed in seeds]
        social_evaluations = [eval_by_trial[name, "social_ppo", seed] for seed in seeds]
        social_early_rows = [row for seed in seeds for row in
                             by_trial[name, "social_ppo", seed][10:100]]
        output[name] = {
            "early_return_baseline": float(np.mean(early["baseline_ppo"])),
            "early_return_social": float(np.mean(early["social_ppo"])),
            "early_return_difference_ci": paired_interval(
                np.array(early["social_ppo"]) - np.array(early["baseline_ppo"])),
            "first100_success_baseline": float(np.mean(success_early["baseline_ppo"])),
            "first100_success_social": float(np.mean(success_early["social_ppo"])),
            "median_90pct_milestone_baseline": float(np.median(milestones["baseline_ppo"])),
            "median_90pct_milestone_social": float(np.median(milestones["social_ppo"])),
            "final_return_baseline": float(np.mean(final["baseline_ppo"])),
            "final_return_social": float(np.mean(final["social_ppo"])),
            "final_return_difference_ci": paired_interval(
                np.array(final["social_ppo"]) - np.array(final["baseline_ppo"])),
            "final_success_baseline": float(np.mean([
                float(eval_by_trial[name, "baseline_ppo", seed]["evaluation_success_rate"])
                for seed in seeds])),
            "final_success_social": float(np.mean([
                float(eval_by_trial[name, "social_ppo", seed]["evaluation_success_rate"])
                for seed in seeds])),
            "median_goal_discovery_episode": float(np.median([
                int(row["discovery_episode"]) for row in social_evaluations])),
            "correct_expert_selections": sum(row["selected_expert"] == "0"
                                             for row in social_evaluations),
            "mean_lambda_episodes_11_100": float(np.mean([
                float(row["coefficient"]) for row in social_early_rows])),
            "fraction_with_expert_match_episodes_11_100": float(np.mean([
                int(row["expert_matches"]) > 0 for row in social_early_rows])),
        }
    return output


def main():
    output = {
        "original": analyze_folder(ROOT / "results"),
        "maze": analyze_folder(ROOT / "results_maze"),
    }
    target = Path(__file__).resolve().parent / "summary_stats.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
