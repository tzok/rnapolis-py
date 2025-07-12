from pycircstat2.clustering import MovM, CircHAC, CircKMeans
from rnapolis import parser_v2 as rna_parser
from rnapolis import tertiary_v2 as tertiary
from pycircstat2 import visualization
from pycircstat2 import descriptive
from pycircstat2 import hypothesis
from pycircstat2 import Circular
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import argparse
import random
import scipy
import math
import sys
import csv


def parse_file(filepath):
    if (filepath[len(filepath)-3:] == "pdb"):
        with open(filepath) as file:
            data = rna_parser.parse_pdb_atoms(file)
    elif (filepath[len(filepath)-3:] == "cif"):
        with open(filepath) as file:
            data = rna_parser.parse_cif_atoms(file)
    else:
        print("Invalid file format.")
        return
    return data


def save_csv(result_file_path, data):
    with open(result_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def dmcd_dmcq(target, model):
    dmcds = []
    dmcqs = []
    angles = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
    for i in range(len(target.index)):
        chain = target.at[i, "chain_id"]
        num = target.at[i, "residue_number"]
        name = target.at[i, "residue_name"]
        if (model.at[i, "chain_id"] == chain and model.at[i, "residue_number"] == num and model.at[i, "residue_name"] == name):
            for angle in angles:
                ang1 = target.at[i, angle]
                ang2 = model.at[i, angle]
                if not (math.isnan(ang1) or math.isnan(ang2)):
                    dmcd = math.atan2(math.sin(ang1 - ang2), math.cos(ang1 - ang2))
                    dmcds.append(dmcd)
                    dmcqs.append(abs(dmcd))
        else:
            for j in range(len(model.index)):
                if (model.at[j, "chain_id"] == chain and model.at[j, "residue_number"] == num and model.at[j, "residue_name"] == name):
                    for angle in angles:
                        ang1 = target.at[i, angle]
                        ang2 = model.at[j, angle]
                        if not (math.isnan(ang1) or math.isnan(ang1)):
                            dmcd = math.atan2(math.sin(ang1 - ang2), math.cos(ang1 - ang2))
                            dmcds.append(dmcd)
                            dmcqs.append(abs(dmcd))
    return dmcds, dmcqs


def auto_dmcd_dmcq(target, model):
    angles = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
    target_angles = target[angles].to_numpy()
    model_angles = model[angles].to_numpy()
    dmcd = descriptive.circ_pairdist(target_angles, model_angles, metric="center")
    dmcq = descriptive.circ_pairdist(target_angles, model_angles, metric="geodesic")
    return dmcd, dmcq


def circular_mad(d, med):
    rs = []
    for v in d:
        r = abs(math.atan2(math.sin(v - med), math.cos(v - med)))
        rs.append(r)
    return np.median(rs).item()


def r_ci(d, reps, alfa = 0.05):
    rs = []
    for i in range(reps):
        n = [random.choice(d) for j in range(len(d))]
        r = descriptive.circ_r(np.array(n))
        rs.append(r)
    lower = np.percentile(rs, alfa/2)
    upper = np.percentile(rs, 100-alfa/2)
    return lower.item(), upper.item()


def circular_mad_ci(d, med, reps, alfa = 0.05):
    rs = []
    for i in range(reps):
        n = [random.choice(d) for j in range(len(d))]
        r = circular_mad(n, med)
        rs.append(r)
    lower = np.percentile(rs, alfa/2)
    upper = np.percentile(rs, 100-alfa/2)
    return lower.item(), upper.item()


def simulation_test(d, mad_o, negatives, reps):
    n = 0
    if(negatives):
        for i in range(reps):
            d_new = [random.uniform(-math.pi, math.pi) for _ in range(len(d))]
            mad_r = circular_mad(d_new, descriptive.circ_median(np.array(d_new)))
            if (mad_r <= mad_o): n += 1
    else:
        for i in range(reps):
            d_new = [random.uniform(0, math.pi) for _ in range(len(d))]
            mad_r = circular_mad(d_new, descriptive.circ_median(np.array(d_new)))
            if (mad_r <= mad_o): n += 1
    return n/reps


def score(d):
    wf = [0.4, 0.6]
    wc = [0.15, 0.3, 0.25, 0.3]
    wu = [0.1, 0.2, 0.1, 0.2, 0.15, 0.25]
    ws = [0.5, 0.2, 0.3]

    f = wf[0] * max(0, 1 - d["mcq"] / (0.5 * math.pi)) + wf[1] * max(0, 1 - d["medcq"] / (0.5 * math.pi))

    c = wc[0] * d["rmcq"] + wc[1] * max(0, 1 - d["circular_mad_mcq"]) + wc[2] * (1 - d["p_watson_dmcq"]) + wc[3] * (1 - d["p_sim_test_dmcq"])
    
    ## to jest zgodnie ze wzorem
    u = wu[0] * max(0, 1 - (abs(d["mcd"]) / (0.25 * math.pi))) + wu[1] * max(0, 1 - (abs(d["medcd"]) / (0.25 * math.pi))) + wu[2] * (1 - d["rmcd"]) + wu[3] * max(0, 1 - (d["circular_mad_mcd"] / (0.25 * math.pi))) + wu[4] * d["p_rayleigh_dmcd"] + wu[5] * d["p_wilcoxon_dmcd"]

    ## to jest do 1
    ##u = wu[0] * max(0, 1 - (abs(d["mcd"]) / (0.25 * math.pi))) + wu[1] * max(0, 1 - (abs(d["medcd"]) / (0.25 * math.pi))) + wu[2] * d["rmcd"] + wu[3] * max(0, 1 - (d["circular_mad_mcd"] / (0.25 * math.pi))) + wu[4] * (1 - d["p_rayleigh_dmcd"]) + wu[5] * (1 - d["p_wilcoxon_dmcd"])

    score = ws[0] * f + ws[1] * c + ws[2] * u
    
    return score


def visualize(d, outfile1, outfile2):

    if np.allclose(d, d[0]):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        ax.set_title("All values equal", pad=30)
        ax.plot([0], [1], 'o')
        plt.savefig(outfile1, format="svg")
        plt.close()
        return

    c = Circular(d, unit="radian")
    ax_labels = ["A", "B"]

    fig, ax = plt.subplot_mosaic(
        mosaic="""
        A
        """,
        figsize=(10, 10), 
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )
    c.plot(ax["A"])
    ax["A"].set_rlim(0, 3)
    ax["A"].set_title("Complete data", pad=30)
    plt.savefig(outfile1, format="svg")
    plt.close()

    movm = MovM(n_clusters=2, unit="radian", random_seed=2046)
    movm.fit(X=d)

    fig, ax = plt.subplot_mosaic(
        mosaic="""
        AB
        """, figsize=(10, 8), 
        subplot_kw={"projection": "polar"},
        layout="constrained",
    )

    for i, k in enumerate([1, 0]):
        x_k = movm.data[movm.labels_ == k]
        c_k = Circular(data=x_k, unit=movm.unit)
        c_k.plot(
            ax=ax[ax_labels[i]],
            config={
                "density": {"color": f"C{i}"},
                "scatter": {"color": f"C{i}"},
                "mean": {"color": f"C{i}"},
                "median": {"color": f"C{i}"},
                "rose": {"color": f"C{i}"},
            }
        )
        ax["A"].set_rlim(0, 3)
        ax["B"].set_rlim(0, 3)

    ax["A"].set_title("Cluster 1", pad=30)
    ax["B"].set_title("Cluster 2", pad=30)
    plt.savefig(outfile2, format="svg")
    plt.close() 



def main(argv):
    parser = argparse.ArgumentParser(description="run scoring")
    parser.add_argument("--target_path", type=str, help="target", required=True) 
    parser.add_argument("--model_path", type=str, help="model", required=True) 
    parser.add_argument("--bootstrap_reps", type=int, help="reps", required=False, default=10000) 
    parser.add_argument("--rounding", type=int, help="rouding", required=False, default=10)
    parser.add_argument("--visualize", type=bool, help="rouding", required=False, default=False)

    args = parser.parse_args()

    target_path = args.target_path
    model_path = args.model_path
    reps = args.bootstrap_reps
    rounding = args.rounding
    visualize_on = args.visualize

    target_data = parse_file(target_path)
    target_structure = tertiary.Structure(target_data)
    target_torsion_angles = target_structure.torsion_angles

    model_data = parse_file(model_path)
    model_structure = tertiary.Structure(model_data)
    model_torsion_angles = model_structure.torsion_angles

    dmcd, dmcq = dmcd_dmcq(target_torsion_angles, model_torsion_angles)
    npdmcd, npdmcq = np.array(dmcd), np.array(dmcq)
    npdmcq_double = np.array([2*v for v in dmcq])

    if(visualize_on):
        visualize(npdmcq, "dmcq.svg", "dmcq_clusters.svg")
        visualize(npdmcd, "dmcd.svg", "dmcd_clusters.svg")

    mcq, rmcq = descriptive.circ_mean_and_r(npdmcq)
    mcd, rmcd = descriptive.circ_mean_and_r(npdmcd)

    if(np.allclose(npdmcd, 0)):
        ci_lower_mcq, ci_upper_mcq, ci_lower_mcd, ci_upper_mcd = [0,0,0,0]
    else:
        ci_lower_mcq, ci_upper_mcq = descriptive.circ_mean_ci(npdmcq, method = "bootstrap", ci = 0.95, mean = mcq)
        ci_lower_mcd, ci_upper_mcd = descriptive.circ_mean_ci(npdmcd, method = "bootstrap", ci = 0.95, mean = mcd)

    ci_lower_rmcq, ci_upper_rmcq = r_ci(dmcq, reps)
    ci_lower_rmcd, ci_upper_rmcd = r_ci(dmcd, reps)

    if(np.allclose(npdmcd, 0)):
        medcq, medcd = [0,0]
    else:
        medcq = descriptive.circ_median(npdmcq) 
        medcd = descriptive.circ_median(npdmcd)

    ci_lower_medcq, ci_upper_medcq = [float(v) for v in descriptive.circ_median_ci(alpha = npdmcq, method = "bootstrap", ci = 0.95, median = medcq)[:2]]
    ci_lower_medcd, ci_upper_medcd = [float(v) for v in descriptive.circ_median_ci(alpha = npdmcd, method = "bootstrap", ci = 0.95, median = medcd)[:2]]

    circular_mad_mcq = circular_mad(dmcq, medcq)
    circular_mad_mcd = circular_mad(dmcd, medcd)

    ci_lower_circular_mad_mcq, ci_upper_circular_mad_mcq = circular_mad_ci(dmcq, medcq, reps)
    ci_lower_circular_mad_mcd, ci_upper_circular_mad_mcd = circular_mad_ci(dmcq, medcd, reps)

    p_rayleigh_dmcd = hypothesis.rayleigh_test(npdmcd).pval.item()
    p_rayleigh_dmcq = hypothesis.rayleigh_test(npdmcq_double).pval.item()

    p_sim_test_dmcd = simulation_test(dmcd, circular_mad_mcd, True, reps)
    p_sim_test_dmcq = simulation_test(dmcq, circular_mad_mcq, False, reps)

    if(np.allclose(npdmcd, 0)):
        p_wilcoxon_dmcd, p_watson_dmcd, p_watson_dmcq  = [0,0,0]
    else:
        p_wilcoxon_dmcd = scipy.stats.wilcoxon(dmcd).pvalue
        p_watson_dmcd = hypothesis.watson_test(npdmcd)[1]
        p_watson_dmcq = hypothesis.watson_test(npdmcq_double)[1]


    results = {
        "mcq": mcq,
        "mcd": mcd,
        "rmcq": rmcq.item(),
        "rmcd": rmcd.item(),
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
        "p_watson_dmcq": p_watson_dmcq
    }

    results["score"] = score(results)

    results_list = [[str(key) for key in results.keys()]] + [[round(v, rounding) for v in list(results.values())]]
    save_csv("result.csv", results_list)
    print(results["score"])


if __name__ == "__main__":
    main(sys.argv[1:])