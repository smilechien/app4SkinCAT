"""Generate the simulation data behind manuscript Tables 3 and 4.

The calibrated item difficulties and empirical ability prior come from
``replay_bundle.zip`` through ``raschcatskin.py``. Each replication simulates
fresh true abilities and item responses; a CAT run and a full-bank estimate use
the same responses for each simulated person.

Example:
    python table34.py --replications 100 --persons-per-level 20

Outputs in ``table34_results/``:
    table3.csv, table4.csv, table3_manuscript.csv,
    table4_manuscript.csv, replication_metrics.csv, summary.csv, protocol.json

The table CSVs contain unrounded numbers so manuscript formatting can be
applied without losing precision. They need not reproduce the values in the
supplied DOCX files unless the simulation settings and random seed match those
used to produce those files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from raschcatskin import BANK, DEFAULT_BUNDLE
from repeated_simulation import (
    LEVELS,
    THRESHOLDS,
    FastRaschEstimator,
    run_replication,
    summarize_replications,
    write_csv,
)


def _write_table(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_tables(summary: list[dict]) -> tuple[list[dict], list[dict]]:
    """Select manuscript columns from equal-weight replication means."""
    by_key = {(row["se_threshold"], row["ability_level"]): row for row in summary}
    table3 = []
    table4 = []
    for threshold in THRESHOLDS:
        overall = by_key[(threshold, "all")]
        table3.append({
            "se_threshold": threshold,
            "mean_cat_length": overall["mean_cat_length_mean"],
            "mean_item_reduction_pct": overall["item_reduction_pct_mean"],
            "mean_cat_full_mae": overall["cat_full_mae_mean"],
            "mean_cat_full_ccc": overall["ccc_mean"],
            "mean_cat_true_mae": overall["cat_true_mae_mean"],
        })
        table4.append({
            "se_threshold": threshold,
            **{level: by_key[(threshold, level)]["cat_true_mae_mean"] for level in LEVELS},
        })
    return table3, table4


def manuscript_tables(table3: list[dict], table4: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply the presentation rounding found in the supplied DOCX tables.

    Their reduction percentages are rounded to one decimal and then displayed
    with two decimal places. The precise percentages remain in table3.csv.
    """
    display3 = []
    for row in table3:
        length = row["mean_cat_length"]
        reduction = row["mean_item_reduction_pct"]
        display3.append({
            "se_threshold": f'{row["se_threshold"]:g}',
            "mean_cat_length": f"{length:.3f}",
            "mean_item_reduction": f"{reduction:.1f}0%",
            "mean_cat_full_mae": f'{row["mean_cat_full_mae"]:.3f}'.rstrip("0").rstrip("."),
            "mean_cat_full_ccc": f'{row["mean_cat_full_ccc"]:.3f}'.rstrip("0").rstrip("."),
            "mean_cat_true_mae": f'{row["mean_cat_true_mae"]:.3f}',
        })
    display4 = [
        {"se_threshold": f'{row["se_threshold"]:g}',
         **{level: f'{row[level]:.3f}' for level in LEVELS}}
        for row in table4
    ]
    return display3, display4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--persons-per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--lower-cut", type=float, default=None)
    parser.add_argument("--upper-cut", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("table34_results"))
    parser.add_argument("--save-matrices", action="store_true",
                        help="Save each simulated full response matrix under output-dir/matrices")
    args = parser.parse_args()

    if args.replications < 2 or args.persons_per_level < 3:
        parser.error("Use at least 2 replications and 3 persons per level")
    if not DEFAULT_BUNDLE.is_file():
        parser.error(f"Required calibrated bundle is missing: {DEFAULT_BUNDLE}")

    z_cut = NormalDist().inv_cdf(2.0 / 3.0)
    lower_cut = (float(args.lower_cut) if args.lower_cut is not None
                 else float(BANK.prior_mean - z_cut * BANK.prior_sd))
    upper_cut = (float(args.upper_cut) if args.upper_cut is not None
                 else float(BANK.prior_mean + z_cut * BANK.prior_sd))
    if not (math.isfinite(lower_cut) and math.isfinite(upper_cut) and lower_cut < upper_cut):
        parser.error("Ability cutpoints must be finite and lower-cut must be below upper-cut")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimator = FastRaschEstimator(float(BANK.prior_mean))
    seeds = np.random.SeedSequence(args.seed).spawn(args.replications)
    rows: list[dict] = []
    for replication, seed in enumerate(seeds, 1):
        matrix_path = (args.output_dir / "matrices" / f"replication_{replication:04d}.csv"
                       if args.save_matrices else None)
        metrics = run_replication(estimator, np.random.default_rng(seed),
                                  args.persons_per_level, lower_cut, upper_cut, matrix_path)
        rows.extend({"replication": replication, **row} for row in metrics)
        if replication == 1 or replication % 10 == 0 or replication == args.replications:
            print(f"Completed replication {replication}/{args.replications}", flush=True)

    summary = summarize_replications(rows)
    table3, table4 = make_tables(summary)
    write_csv(args.output_dir / "replication_metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    _write_table(args.output_dir / "table3.csv", list(table3[0]), table3)
    _write_table(args.output_dir / "table4.csv", list(table4[0]), table4)
    display3, display4 = manuscript_tables(table3, table4)
    _write_table(args.output_dir / "table3_manuscript.csv", list(display3[0]), display3)
    _write_table(args.output_dir / "table4_manuscript.csv", list(display4[0]), display4)
    protocol = {
        "bundle": str(DEFAULT_BUNDLE.resolve()),
        "item_count": len(BANK.items),
        "replications": args.replications,
        "persons_per_level_per_replication": args.persons_per_level,
        "seed": args.seed,
        "se_thresholds": list(THRESHOLDS),
        "ability_cutpoints": {"lower": lower_cut, "upper": upper_cut},
        "true_ability_distribution": "Normal(prior_mean, prior_sd), sampled within each tertile",
        "prior_mean": float(BANK.prior_mean),
        "prior_sd": float(BANK.prior_sd),
        "table_aggregation": "Arithmetic mean of replication-level statistics; equal replication weights",
        "cat_full_reference": "Full-bank posterior theta from the same simulated response vector",
        "cat_true_reference": "Simulated true theta",
        "ccc": "Lin concordance correlation coefficient, computed per replication before averaging",
        "item_reduction_pct": "100 * (1 - mean CAT length / item count)",
        "manuscript_reduction_display": "Precise reduction percentage rounded to 1 decimal, then shown with 2 decimals",
        "limitation": "Item parameters are fixed; calibration uncertainty is not simulated",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Table 3 and Table 4 data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
