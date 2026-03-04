"""Compare RNA 3D structures using circular statistics on torsion angles.

This module implements a scoring pipeline that computes signed (MCD) and unsigned
(MCQ) circular differences between backbone torsion angles of a target and model
RNA structure, then derives a composite similarity score in [0, 1].

Terminology:
    MCD (Mean Circular Deviation): Signed angular difference, range [-pi, pi].
    MCQ (Mean Circular Quality): Unsigned (absolute) angular difference, range [0, pi].
"""

from rnapolis import parser_v2 as rna_parser
from rnapolis import tertiary_v2 as tertiary
from pycircstat2 import descriptive
from pycircstat2 import hypothesis
from pycircstat2 import Circular
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import scipy
import math
import sys
import csv
import os

#: Backbone torsion angle names used for comparison.
TORSION_ANGLES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]

#: Residue identifier columns used to match residues between structures.
RESIDUE_ID_COLUMNS = ["chain_id", "residue_number", "residue_name"]

#: Normalization threshold for MCQ metrics (pi/2 = 90 degrees).
MCQ_NORMALIZATION = 0.5 * math.pi

#: Normalization threshold for MCD metrics (pi/4 = 45 degrees).
MCD_NORMALIZATION = 0.25 * math.pi

#: Weights for the fit sub-score: [mean_mcq, median_mcq].
WEIGHTS_FIT = [0.4, 0.6]

#: Weights for the concentration sub-score: [r_mcq, mad_mcq, watson_p, sim_p].
WEIGHTS_CONCENTRATION = [0.15, 0.3, 0.25, 0.3]

#: Weights for the uniformity sub-score: [mean_mcd, median_mcd, r_mcd, mad_mcd, rayleigh_p, wilcoxon_p].
WEIGHTS_UNIFORMITY = [0.1, 0.2, 0.1, 0.2, 0.15, 0.25]

#: Weights for combining sub-scores: [fit, concentration, uniformity].
WEIGHTS_COMPOSITE = [0.5, 0.2, 0.3]


def parse_file(filepath):
    """Parse an RNA structure file in PDB or mmCIF format.

    Args:
        filepath: Path to the input file. Must have a ``.pdb`` or ``.cif`` extension.

    Returns:
        Parsed atom data as a DataFrame.

    Raises:
        ValueError: If the file extension is not ``.pdb`` or ``.cif``.
    """
    if filepath.endswith(".pdb"):
        with open(filepath) as file:
            data = rna_parser.parse_pdb_atoms(file)
    elif filepath.endswith(".cif"):
        with open(filepath) as file:
            data = rna_parser.parse_cif_atoms(file)
    else:
        raise ValueError(
            f"Invalid file format for '{filepath}'. Expected .pdb or .cif extension."
        )
    return data


def save_csv(result_file_path, data):
    """Write rows of data to a CSV file.

    Args:
        result_file_path: Path to the output CSV file.
        data: List of rows, where each row is a list of values.
    """
    with open(result_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def _compute_angle_diffs(target_row, model_row):
    """Compute wrapped angular differences for all torsion angles between two residues."""
    signed = []
    unsigned = []
    for angle in TORSION_ANGLES:
        target_angle = target_row[angle]
        model_angle = model_row[angle]
        if not (math.isnan(target_angle) or math.isnan(model_angle)):
            wrapped_diff = math.atan2(
                math.sin(target_angle - model_angle),
                math.cos(target_angle - model_angle),
            )
            if not math.isnan(wrapped_diff):
                signed.append(wrapped_diff)
                unsigned.append(abs(wrapped_diff))
    return signed, unsigned


def _build_model_lookup(model):
    """Build a dict mapping residue identity to row index for O(1) lookup."""
    lookup = {}
    for j in range(len(model)):
        row = model.iloc[j]
        key = (row["chain_id"], row["residue_number"], row["residue_name"])
        lookup[key] = j
    return lookup


def compute_mcd_mcq(target, model):
    """Compute signed (MCD) and unsigned (MCQ) angular differences between two structures.

    For each residue matched by chain, number, and name, wraps the angular
    difference of each backbone torsion angle into [-pi, pi] (MCD) and takes
    its absolute value (MCQ).

    If a residue at position *i* in the target does not match the same position
    in the model, a lookup by residue identity is used as a fallback.

    Args:
        target: DataFrame of torsion angles for the reference structure.
        model: DataFrame of torsion angles for the model structure.

    Returns:
        Tuple of (signed_diffs, unsigned_diffs) where each is a list of
        floats in radians.
    """
    signed_diffs = []
    unsigned_diffs = []
    model_lookup = _build_model_lookup(model)

    for i in range(len(target)):
        target_row = target.iloc[i]
        chain_id = target_row["chain_id"]
        residue_number = target_row["residue_number"]
        residue_name = target_row["residue_name"]

        if (
            i < len(model)
            and model.iloc[i]["chain_id"] == chain_id
            and model.iloc[i]["residue_number"] == residue_number
            and model.iloc[i]["residue_name"] == residue_name
        ):
            s, u = _compute_angle_diffs(target_row, model.iloc[i])
            signed_diffs.extend(s)
            unsigned_diffs.extend(u)
        else:
            key = (chain_id, residue_number, residue_name)
            if key in model_lookup:
                j = model_lookup[key]
                s, u = _compute_angle_diffs(target_row, model.iloc[j])
                signed_diffs.extend(s)
                unsigned_diffs.extend(u)

    return signed_diffs, unsigned_diffs


def circular_mad(data, median):
    """Compute the circular median absolute deviation.

    Args:
        data: Angular values in radians.
        median: Circular median in radians.

    Returns:
        The median of absolute wrapped deviations from ``median``.
    """
    data = np.asarray(data)
    deviations = np.abs(np.arctan2(np.sin(data - median), np.cos(data - median)))
    return float(np.median(deviations))


def bootstrap_ci(data, statistic_fn, n_bootstrap, alpha=0.05):
    """Compute a bootstrap confidence interval for a circular statistic.

    Args:
        data: Input data (list of floats).
        statistic_fn: Callable that takes a list and returns a scalar.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
    """
    statistics = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True).tolist()
        statistics.append(statistic_fn(sample))
    lower = np.percentile(statistics, 100 * alpha / 2)
    upper = np.percentile(statistics, 100 - 100 * alpha / 2)
    return float(lower), float(upper)


def monte_carlo_mad_test(data, observed_mad, use_full_circle, n_simulations):
    """Test concentration by comparing observed MAD to random circular samples.

    Generates ``n_simulations`` random samples from a uniform circular
    distribution and counts how often their MAD is at most as small as
    the ``observed_mad``.

    Args:
        data: Original angular data (used only for its length).
        observed_mad: The observed circular MAD to compare against.
        use_full_circle: If ``True``, sample from [-pi, pi]; otherwise [0, pi].
        n_simulations: Number of Monte Carlo iterations.

    Returns:
        Proportion of random samples with MAD <= ``observed_mad`` (p-value).
    """
    lower = -math.pi if use_full_circle else 0
    count = 0
    for _ in range(n_simulations):
        random_sample = [random.uniform(lower, math.pi) for _ in range(len(data))]
        random_mad = circular_mad(
            random_sample, descriptive.circ_median(np.array(random_sample))
        )
        if random_mad <= observed_mad:
            count += 1
    return count / n_simulations


def compute_score(metrics):
    """Compute a composite similarity score from circular statistics.

    Combines three weighted sub-scores:

    - **Fit**: How close the mean and median MCQ are to zero.
    - **Concentration**: How tightly the MCQ distribution is concentrated.
    - **Uniformity**: How concentrated the MCD distribution is around zero,
      indicating lack of systematic bias.

    Args:
        metrics: Dictionary containing at minimum the keys ``mcq``, ``medcq``,
            ``rmcq``, ``circular_mad_mcq``, ``p_watson_dmcq``, ``p_sim_test_dmcq``,
            ``mcd``, ``medcd``, ``rmcd``, ``circular_mad_mcd``,
            ``p_rayleigh_dmcd``, ``p_wilcoxon_dmcd``.

    Returns:
        Similarity score in [0, 1], where 1.0 means identical structures.
    """
    wf = WEIGHTS_FIT
    wc = WEIGHTS_CONCENTRATION
    wu = WEIGHTS_UNIFORMITY
    ws = WEIGHTS_COMPOSITE

    fit = wf[0] * max(0, 1 - metrics["mcq"] / MCQ_NORMALIZATION) + wf[1] * max(
        0, 1 - metrics["medcq"] / MCQ_NORMALIZATION
    )

    concentration = (
        wc[0] * metrics["rmcq"]
        + wc[1] * max(0, 1 - metrics["circular_mad_mcq"])
        + wc[2] * (1 - metrics["p_watson_dmcq"])
        + wc[3] * (1 - metrics["p_sim_test_dmcq"])
    )

    uniformity = (
        wu[0] * max(0, 1 - (abs(metrics["mcd"]) / MCD_NORMALIZATION))
        + wu[1] * max(0, 1 - (abs(metrics["medcd"]) / MCD_NORMALIZATION))
        + wu[2] * metrics["rmcd"]
        + wu[3] * max(0, 1 - (metrics["circular_mad_mcd"] / MCD_NORMALIZATION))
        + wu[4] * (1 - metrics["p_rayleigh_dmcd"])
        + wu[5] * (1 - metrics["p_wilcoxon_dmcd"])
    )

    composite = ws[0] * fit + ws[1] * concentration + ws[2] * uniformity

    return composite


def _build_plot_config(c, color="black"):
    """Build an adaptive pycircstat2 plot config based on data concentration.

    For highly concentrated data (``r > 0.9``), nonparametric density estimation
    produces an extreme spike that dominates the plot.  In that regime the density
    layer is disabled and the rose diagram uses finer bins so that the angular
    distribution remains readable.

    Args:
        c: A ``pycircstat2.Circular`` object whose ``r`` attribute is inspected.
        color: Color string applied to scatter, mean, median, and rose elements.

    Returns:
        Configuration dictionary suitable for ``Circular.plot(config=...)``.
    """
    high_concentration = float(c.r) > 0.9

    config = {
        "scatter": {"color": color},
        "mean": {"color": color},
        "median": {"color": color},
        "rose": {"color": color},
    }

    if high_concentration:
        config["density"] = False
        config["rose"]["bins"] = 36
    else:
        config["density"] = {"color": color}

    return config


def visualize(d, outfile):
    """Save a polar plot of angular data to an SVG file.

    Plots the circular distribution of angle differences using an adaptive
    configuration: for highly concentrated data (resultant length > 0.9) the
    density layer is disabled to avoid an uninformative spike, and the rose
    diagram uses finer bins.

    Input angles are wrapped to [0, 2pi) before plotting so that signed
    differences (which may be negative) do not confuse the density estimator
    or scatter positioning.

    Args:
        d: Array of angular values in radians.
        outfile: Output path for the polar plot (typically ``.svg``).
    """
    d = d % (2 * np.pi)

    if np.allclose(d, d[0]):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        ax.set_title("All values equal", pad=30)
        ax.plot([0], [1], "o")
        plt.savefig(outfile, format="svg")
        plt.close()
        return

    c = Circular(d, unit="radian")
    config = _build_plot_config(c)

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    c.plot(ax, config=config)
    ax.set_title("Complete data", pad=30)
    plt.savefig(outfile, format="svg")
    plt.close()


def _parse_structures(target_path, model_path):
    """Parse two structure files and extract torsion angle DataFrames."""
    target_data = parse_file(target_path)
    target_structure = tertiary.Structure(target_data)
    target_torsion = target_structure.torsion_angles

    model_data = parse_file(model_path)
    model_structure = tertiary.Structure(model_data)
    model_torsion = model_structure.torsion_angles

    return target_torsion, model_torsion


def _compute_statistics(signed_diffs, unsigned_diffs, n_bootstrap):
    """Compute all circular statistics, CIs, and hypothesis tests."""
    signed_diffs_arr = np.array(signed_diffs)
    unsigned_diffs_arr = np.array(unsigned_diffs)
    doubled_unsigned_arr = 2 * unsigned_diffs_arr

    all_diffs_zero = np.allclose(signed_diffs_arr, 0)

    mcq, rmcq = descriptive.circ_mean_and_r(unsigned_diffs_arr)
    mcd, rmcd = descriptive.circ_mean_and_r(signed_diffs_arr)

    if all_diffs_zero:
        ci_lower_mcq, ci_upper_mcq, ci_lower_mcd, ci_upper_mcd = 0, 0, 0, 0
    else:
        ci_lower_mcq, ci_upper_mcq = descriptive.circ_mean_ci(
            unsigned_diffs_arr, method="bootstrap", ci=0.95, mean=mcq
        )
        ci_lower_mcd, ci_upper_mcd = descriptive.circ_mean_ci(
            signed_diffs_arr, method="bootstrap", ci=0.95, mean=mcd
        )

    ci_lower_rmcq, ci_upper_rmcq = bootstrap_ci(
        unsigned_diffs, lambda s: descriptive.circ_r(np.array(s)), n_bootstrap
    )
    ci_lower_rmcd, ci_upper_rmcd = bootstrap_ci(
        signed_diffs, lambda s: descriptive.circ_r(np.array(s)), n_bootstrap
    )

    if all_diffs_zero:
        medcq, medcd = 0, 0
    else:
        medcq = descriptive.circ_median(unsigned_diffs_arr)
        medcd = descriptive.circ_median(signed_diffs_arr)

    ci_lower_medcq, ci_upper_medcq = [
        float(v)
        for v in descriptive.circ_median_ci(
            alpha=unsigned_diffs_arr, method="bootstrap", ci=0.95, median=medcq
        )[:2]
    ]
    ci_lower_medcd, ci_upper_medcd = [
        float(v)
        for v in descriptive.circ_median_ci(
            alpha=signed_diffs_arr, method="bootstrap", ci=0.95, median=medcd
        )[:2]
    ]

    circular_mad_mcq = circular_mad(unsigned_diffs, medcq)
    circular_mad_mcd = circular_mad(signed_diffs, medcd)

    ci_lower_circular_mad_mcq, ci_upper_circular_mad_mcq = bootstrap_ci(
        unsigned_diffs, lambda s: circular_mad(s, medcq), n_bootstrap
    )
    ci_lower_circular_mad_mcd, ci_upper_circular_mad_mcd = bootstrap_ci(
        signed_diffs, lambda s: circular_mad(s, medcd), n_bootstrap
    )

    p_rayleigh_dmcd = float(hypothesis.rayleigh_test(signed_diffs_arr).pval)
    p_rayleigh_dmcq = float(hypothesis.rayleigh_test(doubled_unsigned_arr).pval)

    p_sim_test_dmcd = monte_carlo_mad_test(
        signed_diffs, circular_mad_mcd, True, n_bootstrap
    )
    p_sim_test_dmcq = monte_carlo_mad_test(
        unsigned_diffs, circular_mad_mcq, False, n_bootstrap
    )

    if all_diffs_zero:
        p_wilcoxon_dmcd, p_watson_dmcd, p_watson_dmcq = 0, 0, 0
    else:
        p_wilcoxon_dmcd = scipy.stats.wilcoxon(signed_diffs).pvalue
        p_watson_dmcd = hypothesis.watson_test(signed_diffs_arr).pval
        p_watson_dmcq = hypothesis.watson_test(doubled_unsigned_arr).pval

    return {
        "mcq": mcq,
        "mcd": mcd,
        "rmcq": float(rmcq),
        "rmcd": float(rmcd),
        "ci_lower_mcq": ci_lower_mcq,
        "ci_upper_mcq": ci_upper_mcq,
        "ci_lower_mcd": ci_lower_mcd,
        "ci_upper_mcd": ci_upper_mcd,
        "ci_lower_rmcq": ci_lower_rmcq,
        "ci_upper_rmcq": ci_upper_rmcq,
        "ci_lower_rmcd": ci_lower_rmcd,
        "ci_upper_rmcd": ci_upper_rmcd,
        "medcq": medcq,
        "medcd": medcd,
        "ci_lower_medcq": ci_lower_medcq,
        "ci_upper_medcq": ci_upper_medcq,
        "ci_lower_medcd": ci_lower_medcd,
        "ci_upper_medcd": ci_upper_medcd,
        "circular_mad_mcq": circular_mad_mcq,
        "circular_mad_mcd": circular_mad_mcd,
        "ci_lower_circular_mad_mcq": ci_lower_circular_mad_mcq,
        "ci_upper_circular_mad_mcq": ci_upper_circular_mad_mcq,
        "ci_lower_circular_mad_mcd": ci_lower_circular_mad_mcd,
        "ci_upper_circular_mad_mcd": ci_upper_circular_mad_mcd,
        "p_rayleigh_dmcd": p_rayleigh_dmcd,
        "p_rayleigh_dmcq": p_rayleigh_dmcq,
        "p_sim_test_dmcd": p_sim_test_dmcd,
        "p_sim_test_dmcq": p_sim_test_dmcq,
        "p_wilcoxon_dmcd": p_wilcoxon_dmcd,
        "p_watson_dmcd": p_watson_dmcd,
        "p_watson_dmcq": p_watson_dmcq,
    }


def evaluate_similarity(
    target_path, model_path, n_bootstrap=10000, visualize_on=False, output_dir="."
):
    """Evaluate the similarity between two RNA structures.

    Parses both structures, computes torsion angle differences, derives
    circular statistics with bootstrap confidence intervals, and returns
    a results dictionary including a composite score.

    Args:
        target_path: Path to the reference structure (``.pdb`` or ``.cif``).
        model_path: Path to the model structure (``.pdb`` or ``.cif``).
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        visualize_on: If ``True``, save polar plots to ``output_dir``.
        output_dir: Directory for output files (plots, CSV).

    Returns:
        Dictionary of circular statistics and a ``score`` key with the
        composite similarity score in [0, 1].
    """
    target_torsion, model_torsion = _parse_structures(target_path, model_path)

    signed_diffs, unsigned_diffs = compute_mcd_mcq(target_torsion, model_torsion)

    if visualize_on:
        unsigned_diffs_arr = np.array(unsigned_diffs)
        signed_diffs_arr = np.array(signed_diffs)
        visualize(
            unsigned_diffs_arr,
            os.path.join(output_dir, "dmcq.svg"),
        )
        visualize(
            signed_diffs_arr,
            os.path.join(output_dir, "dmcd.svg"),
        )

    results = _compute_statistics(signed_diffs, unsigned_diffs, n_bootstrap)
    results["score"] = compute_score(results)

    return results


def main():
    """CLI entry point for RNA structure scoring."""
    parser = argparse.ArgumentParser(
        description="Compare two RNA 3D structures using circular statistics"
    )
    parser.add_argument(
        "--target", type=str, help="path to reference structure", required=True
    )
    parser.add_argument(
        "--model", type=str, help="path to model structure", required=True
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        help="number of bootstrap resamples",
        required=False,
        default=10000,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save polar plots to output directory",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="output directory for CSV and plots",
        required=False,
        default=".",
    )

    args = parser.parse_args()

    results = evaluate_similarity(
        args.target,
        args.model,
        args.bootstrap_reps,
        args.visualize,
        args.output_dir,
    )

    results_list = [[str(key) for key in results.keys()]] + [
        list(results.values())
    ]
    save_csv(os.path.join(args.output_dir, "result.csv"), results_list)
    print(results["score"])


if __name__ == "__main__":
    main()
