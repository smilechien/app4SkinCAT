"""Independent Rasch-CAT simulation replications for method-agreement review.

Run offline: python repeated_simulation.py --replications 100 --persons-per-level 20
Each replication generates a new response matrix from the calibrated item bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from raschcatskin import BANK, _trapz_compat


THRESHOLDS = (0.25, 0.32, 0.40)
LEVELS = ("lower", "middle", "upper")


def _posterior_summary(log_post: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    weights = np.exp(log_post - np.max(log_post))
    weights /= _trapz_compat(weights, grid)
    mean = float(_trapz_compat(grid * weights, grid))
    variance = float(_trapz_compat((grid - mean) ** 2 * weights, grid))
    return mean, max(math.sqrt(max(variance, 1e-10)), 1e-6)


class FastRaschEstimator:
    """Posterior updates equivalent to the app, without refitting prior responses."""

    def __init__(self, start_theta: float):
        self.grid = np.asarray(BANK.theta_grid, dtype=float)
        self.items = list(BANK.items)
        self.deltas = np.asarray([item.delta for item in self.items], dtype=float)
        self.item_nos = np.asarray([item.no for item in self.items], dtype=int)
        prior_sd = float(BANK.prior_sd)
        self.prior_log = (-0.5 * ((self.grid - start_theta) / prior_sd) ** 2
                          - math.log(prior_sd * math.sqrt(2.0 * math.pi)))
        probabilities = np.clip(BANK.probability(self.grid[None, :], self.deltas[:, None]),
                                1e-12, 1.0 - 1e-12)
        self.logp = np.log(probabilities)
        self.log1mp = np.log1p(-probabilities)
        self.full_base = self.prior_log + np.sum(self.log1mp, axis=0)
        self.full_score_effect = self.logp - self.log1mp

    def full_theta(self, scores: np.ndarray) -> float:
        log_post = self.full_base + scores @ self.full_score_effect
        return _posterior_summary(log_post, self.grid)[0]

    def cat_at_thresholds(self, scores: np.ndarray, first_item_index: int) -> dict[float, tuple[int, float]]:
        used = np.zeros(len(self.items), dtype=bool)
        log_post = self.prior_log.copy()
        theta, se = _posterior_summary(log_post, self.grid)
        results: dict[float, tuple[int, float]] = {}
        for length in range(1, len(self.items) + 1):
            if length == 1:
                item_index = first_item_index
            else:
                remaining = np.flatnonzero(~used)
                probabilities = 1.0 / (1.0 + np.exp(np.clip(-(theta - self.deltas[remaining]), -35, 35)))
                information = probabilities * (1.0 - probabilities)
                item_index = int(max(
                    zip(remaining, information),
                    key=lambda pair: (float(pair[1]), -abs(float(self.deltas[pair[0]] - theta)),
                                      -int(self.item_nos[pair[0]])),
                )[0])
            used[item_index] = True
            log_post += self.logp[item_index] if scores[item_index] == 1 else self.log1mp[item_index]
            theta, se = _posterior_summary(log_post, self.grid)
            for threshold in THRESHOLDS:
                if threshold not in results and se <= threshold:
                    results[threshold] = (length, theta)
            if len(results) == len(THRESHOLDS):
                break
        for threshold in THRESHOLDS:
            results.setdefault(threshold, (length, theta))
        return results


def _sample_true_theta(rng: np.random.Generator, level: str,
                       lower_cut: float, upper_cut: float) -> float:
    while True:
        theta = float(rng.normal(BANK.prior_mean, BANK.prior_sd))
        if ((level == "lower" and theta < lower_cut)
                or (level == "middle" and lower_cut <= theta <= upper_cut)
                or (level == "upper" and theta > upper_cut)):
            return theta


def _mean_and_rmse(errors: np.ndarray) -> dict:
    return {
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def _agreement(cat_theta: np.ndarray, full_theta: np.ndarray) -> dict:
    differences = cat_theta - full_theta
    cat_centered = cat_theta - np.mean(cat_theta)
    full_centered = full_theta - np.mean(full_theta)
    denominator = float(np.mean(cat_centered ** 2) + np.mean(full_centered ** 2)
                        + (np.mean(cat_theta) - np.mean(full_theta)) ** 2)
    ccc = float(2.0 * np.mean(cat_centered * full_centered) / denominator) if denominator > 1e-12 else float("nan")
    difference_sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    bias = float(np.mean(differences))
    return {
        "ccc": ccc,
        "loa_low": bias - 1.96 * difference_sd,
        "loa_high": bias + 1.96 * difference_sd,
    }


def run_replication(estimator: FastRaschEstimator, rng: np.random.Generator,
                    persons_per_level: int, lower_cut: float, upper_cut: float,
                    matrix_path: Path | None = None) -> list[dict]:
    persons = []
    for level in LEVELS:
        for _ in range(persons_per_level):
            true_theta = _sample_true_theta(rng, level, lower_cut, upper_cut)
            probabilities = BANK.probability(true_theta, estimator.deltas)
            scores = (rng.random(len(estimator.items)) < probabilities).astype(np.int8)
            full_theta = estimator.full_theta(scores)
            first_item = int(rng.integers(len(estimator.items)))
            cat_results = estimator.cat_at_thresholds(scores, first_item)
            persons.append((level, true_theta, full_theta, scores, cat_results))
    if matrix_path is not None:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        with matrix_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["person_id", "ability_level", "true_theta", "full_theta"]
                            + [item.item_id for item in estimator.items])
            for index, (level, true_theta, full_theta, scores, _) in enumerate(persons, 1):
                writer.writerow([index, level, true_theta, full_theta] + scores.tolist())
    true_thetas = np.asarray([person[1] for person in persons])
    full_thetas = np.asarray([person[2] for person in persons])
    levels = np.asarray([person[0] for person in persons])
    rows = []
    for threshold in THRESHOLDS:
        lengths = np.asarray([person[4][threshold][0] for person in persons], dtype=float)
        cat_thetas = np.asarray([person[4][threshold][1] for person in persons])
        agreement = _agreement(cat_thetas, full_thetas)
        for level in ("all",) + LEVELS:
            mask = np.ones(len(persons), dtype=bool) if level == "all" else levels == level
            cat_full = _mean_and_rmse(cat_thetas[mask] - full_thetas[mask])
            cat_true = _mean_and_rmse(cat_thetas[mask] - true_thetas[mask])
            selected_lengths = lengths[mask]
            row = {
                "se_threshold": threshold,
                "ability_level": level,
                "n": int(np.sum(mask)),
                "mean_cat_length": float(np.mean(selected_lengths)),
                "median_cat_length": float(np.median(selected_lengths)),
                "q1_cat_length": float(np.percentile(selected_lengths, 25)),
                "q3_cat_length": float(np.percentile(selected_lengths, 75)),
                "item_reduction_pct": float(100.0 * (1.0 - np.mean(selected_lengths) / len(estimator.items))),
                "cat_full_bias": cat_full["bias"],
                "cat_full_mae": cat_full["mae"],
                "cat_full_rmse": cat_full["rmse"],
                "cat_true_bias": cat_true["bias"],
                "cat_true_mae": cat_true["mae"],
                "cat_true_rmse": cat_true["rmse"],
                "ccc": agreement["ccc"] if level == "all" else _agreement(cat_thetas[mask], full_thetas[mask])["ccc"],
                "loa_low": agreement["loa_low"] if level == "all" else _agreement(cat_thetas[mask], full_thetas[mask])["loa_low"],
                "loa_high": agreement["loa_high"] if level == "all" else _agreement(cat_thetas[mask], full_thetas[mask])["loa_high"],
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_replications(rows: list[dict]) -> list[dict]:
    keys = [key for key in rows[0] if key not in {"replication", "se_threshold", "ability_level", "n"}]
    summary = []
    for threshold in THRESHOLDS:
        for level in ("all",) + LEVELS:
            group = [row for row in rows if row["se_threshold"] == threshold and row["ability_level"] == level]
            out = {"se_threshold": threshold, "ability_level": level,
                   "replications": len(group), "persons_per_replication": group[0]["n"]}
            for key in keys:
                values = np.asarray([row[key] for row in group], dtype=float)
                finite = values[np.isfinite(values)]
                out[key + "_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                out[key + "_mc_se"] = float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else float("nan")
                out[key + "_p2_5"] = float(np.percentile(finite, 2.5)) if len(finite) else float("nan")
                out[key + "_p97_5"] = float(np.percentile(finite, 97.5)) if len(finite) else float("nan")
            summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--persons-per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lower-cut", type=float, help="Upper true-theta bound for the lower ability level")
    parser.add_argument("--upper-cut", type=float, help="Lower true-theta bound for the upper ability level")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("simulation_results"))
    parser.add_argument("--save-matrices", action="store_true")
    args = parser.parse_args()
    if args.replications < 2 or args.persons_per_level < 3:
        parser.error("Use at least 2 replications and 3 persons per level")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_theta = float(BANK.prior_mean)
    z_cut = NormalDist().inv_cdf(2.0 / 3.0)
    lower_cut = float(args.lower_cut if args.lower_cut is not None
                      else BANK.prior_mean - z_cut * BANK.prior_sd)
    upper_cut = float(args.upper_cut if args.upper_cut is not None
                      else BANK.prior_mean + z_cut * BANK.prior_sd)
    if not (math.isfinite(lower_cut) and math.isfinite(upper_cut) and lower_cut < upper_cut):
        parser.error("Ability cutpoints must be finite and lower-cut must be below upper-cut")
    estimator = FastRaschEstimator(start_theta)
    seeds = np.random.SeedSequence(args.seed).spawn(args.replications)
    rows = []
    for replication, seed in enumerate(seeds, 1):
        rng = np.random.default_rng(seed)
        matrix_path = args.output_dir / "matrices" / f"replication_{replication:04d}.csv" if args.save_matrices else None
        run_rows = run_replication(estimator, rng, args.persons_per_level,
                                   lower_cut, upper_cut, matrix_path)
        rows.extend([{"replication": replication, **row} for row in run_rows])
        if replication == 1 or replication % 10 == 0 or replication == args.replications:
            print(f"Completed replication {replication}/{args.replications}", flush=True)
    write_csv(args.output_dir / "replication_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summarize_replications(rows))
    protocol = {
        "replications": args.replications,
        "persons_per_level_per_replication": args.persons_per_level,
        "seed": args.seed,
        "se_thresholds": THRESHOLDS,
        "ability_levels": {
            "lower": f"true theta < {lower_cut:.6f}",
            "middle": f"{lower_cut:.6f} <= true theta <= {upper_cut:.6f}",
            "upper": f"true theta > {upper_cut:.6f}",
        },
        "latent_theta_distribution": f"N({BANK.prior_mean:.6f}, {BANK.prior_sd:.6f}^2), sampled within each level",
        "responses": "Independent Bernoulli draws from calibrated Rasch item probabilities for every person and replication",
        "pairing": "Each person uses one full response vector and the same randomized first CAT item at all SE thresholds",
        "full_reference": "Full-bank posterior theta from all simulated item responses",
        "accuracy_reference": "Known simulated true theta",
        "ccc": "Lin concordance correlation coefficient of paired CAT and full-bank theta estimates",
        "limits_of_agreement": "Mean(CAT-full) +/- 1.96 times sample SD(CAT-full), per replication",
        "summary_intervals": "2.5th and 97.5th percentiles across independent replication-level statistics",
        "limitation": "Item parameters are held fixed; calibration uncertainty is not simulated",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
