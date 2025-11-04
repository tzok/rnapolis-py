import logging
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    Stacking,
    StackingTopology,
)
from rnapolis.tertiary import Structure3D

_BARNABA_STACKING_TOPOLOGIES = {
    ">>": "upward",
    "<<": "downward",
    "<>": "outward",
    "><": "inward",
}


def _barnaba_assign_indices_to_chains(indices, chains, score, avail):
    # Convert to numpy for Hungarian; fall back to greedy if SciPy not available
    S = np.array(score, dtype=float)
    A = np.array(avail, dtype=float)
    # small tie-breaker: prefer chains that have the residues at all
    S2 = S + 1e-6 * A
    # Maximize S2 by minimizing -S2
    row_ind, col_ind = linear_sum_assignment(-S2)

    mapping = {}
    pair_scores = {}
    for i, j in zip(row_ind, col_ind):
        mapping[indices[i]] = chains[j]
        pair_scores[(indices[i], chains[j])] = int(score[i][j])
    return mapping, pair_scores


def _barnaba_map_barnaba_to_rnapolis(
    barnaba: Iterable[Tuple[str, int, int]],
    rnapolis: Iterable[Tuple[str, int, str]],
    name_eq: Callable[[str, str], bool] = lambda a, b: a == b,
):
    # Index input
    barnaba = list(barnaba)
    rnapolis = list(rnapolis)

    # rnapolis grouped by number, and fast lookup by (number, chain)
    rnap_by_num = defaultdict(dict)  # number -> {chain: name}
    rnap_by_num_chain = {}  # (number, chain) -> (name, number, chain)
    chains_all = set()
    for rname, num, chain in rnapolis:
        rnap_by_num[num][chain] = rname
        rnap_by_num_chain[(num, chain)] = (rname, num, chain)
        chains_all.add(chain)

    indices = sorted({idx for _, _, idx in barnaba})
    chains = sorted(chains_all)
    idx_to_row = {idx: i for i, idx in enumerate(indices)}
    ch_to_col = {ch: j for j, ch in enumerate(chains)}

    # Build score and availability matrices
    # score[i, c] = number of name matches if index i -> chain c
    # avail[i, c] = number of residues where index i has a number that exists in chain c
    m, n = len(indices), len(chains)
    score = [[0] * n for _ in range(m)]
    avail = [[0] * n for _ in range(m)]
    for bname, num, idx in barnaba:
        row = idx_to_row[idx]
        for ch, rname in rnap_by_num.get(num, {}).items():
            col = ch_to_col[ch]
            avail[row][col] += 1
            if name_eq(bname, rname):
                score[row][col] += 1

    # Choose assignment that maximizes score (with tiny tie-break on availability)
    mapping, pair_scores = _barnaba_assign_indices_to_chains(
        indices, chains, score, avail
    )

    # Use mapping to pair each barnaba residue to its rnapolis counterpart
    pairs = []
    mismatches = []
    missing = []
    for bname, num, idx in barnaba:
        ch = mapping[idx]
        r = rnap_by_num_chain.get((num, ch))
        if r is None:
            missing.append((bname, num, idx))
            continue
        match = bool(name_eq(bname, r[0]))
        pairs.append(((bname, num, idx), r, match))
        if not match:
            mismatches.append(((bname, num, idx), r))

    # Optional: extras present only in rnapolis
    barnaba_keys = {(num, mapping[idx]) for _, num, idx in barnaba}
    extras_in_rnapolis = [r for r in rnapolis if (r[1], r[2]) not in barnaba_keys]

    return {
        "index_to_chain": mapping,  # dict: barnaba_index -> rnapolis_chain
        "pair_scores": pair_scores,  # per (index, chain) chosen, how many name matches
        "pairs": pairs,  # list of ((bname,num,idx), (rname,num,chain), name_match_bool)
        "mismatches": mismatches,  # list of pairs where names differ
        "missing_in_rnapolis": missing,  # barnaba residues lacking a counterpart (should be empty)
        "extras_in_rnapolis": extras_in_rnapolis,
    }


def _barnaba_get_leontis_westhof(interaction: str) -> Optional[LeontisWesthof]:
    if "x" in interaction.lower():
        return None
    if interaction in ("WCc", "GUc"):
        return LeontisWesthof.cWW
    return LeontisWesthof[f"{interaction[2]}{interaction[:2]}"]


def _barnaba_get_residue(
    residue_info: str,
    barnaba_mapping: Dict,
    residue_mapping: Dict,
    rnapolis_mapping: Dict,
) -> Optional[Residue]:
    barnaba_tuple = barnaba_mapping.get(residue_info, None)
    rnapolis_tuple = residue_mapping.get(barnaba_tuple, None)
    return rnapolis_mapping.get(rnapolis_tuple, None)


def parse_barnaba_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse barnaba output files and convert to BaseInteractions.
    Args:
        file_paths: List of paths to barnaba output files (pairing and stacking)
        structure3d: The 3D structure parsed from PDB/mmCIF
    Returns:
        BaseInteractions object containing the interactions found by barnaba
    """
    pairing_file = None
    stacking_file = None

    for file_path in file_paths:
        if "pairing" in os.path.basename(file_path):
            pairing_file = file_path
        elif "stacking" in os.path.basename(file_path):
            stacking_file = file_path

    if pairing_file is None and stacking_file is None:
        logging.warning("No barnaba pairing or stacking files found")
        return BaseInteractions([], [], [], [], [])

    barnaba_mapping: List[str] = []

    with open(pairing_file or stacking_file, "r") as f:
        for line in f.readlines():
            if line.startswith("# sequence"):
                barnaba_mapping = line.strip().split()[2].split("-")
                break

    if not barnaba_mapping:
        logging.warning("Could not find barnaba sequence in output files")
        return BaseInteractions([], [], [], [], [])

    barnaba_mapping = {
        residue_info: (
            residue_info.split("_")[0],
            int(residue_info.split("_")[1]),
            int(residue_info.split("_")[2]),
        )
        for residue_info in barnaba_mapping
    }
    rnapolis_mapping = {
        (residue.auth.name, residue.auth.number, residue.auth.chain): residue
        for residue in structure3d.residues
        if residue.auth and residue.is_nucleotide
    }
    barnaba_to_rnapolis_mapping = _barnaba_map_barnaba_to_rnapolis(
        barnaba_mapping.values(), rnapolis_mapping.keys()
    )
    residue_mapping = {
        barnaba: rnapolis
        for barnaba, rnapolis, _ in barnaba_to_rnapolis_mapping["pairs"]
    }

    base_pairs: List[BasePair] = []
    stackings: List[Stacking] = []
    other_interactions: List[OtherInteraction] = []

    for file_path, is_pairing, is_stacking in [
        (pairing_file, True, False),
        (stacking_file, False, True),
    ]:
        if file_path is None:
            continue

        logging.info(f"Processing barnaba file: {file_path}")

        with open(file_path) as f:
            content = f.read()

        for line in content.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            fields = line.split()
            if len(fields) < 3:
                continue

            res1_str, res2_str, interaction_str = fields[0], fields[1], fields[2]

            nt1 = _barnaba_get_residue(
                res1_str, barnaba_mapping, residue_mapping, rnapolis_mapping
            )
            nt2 = _barnaba_get_residue(
                res2_str, barnaba_mapping, residue_mapping, rnapolis_mapping
            )

            if not nt1 or not nt2:
                continue

            if is_pairing:
                try:
                    lw = _barnaba_get_leontis_westhof(interaction_str)
                    if lw:
                        base_pairs.append(BasePair(nt1, nt2, lw, None))
                    else:
                        other_interactions.append(OtherInteraction(nt1, nt2))
                except (KeyError, IndexError):
                    other_interactions.append(OtherInteraction(nt1, nt2))
            elif is_stacking:
                try:
                    topology_str = _BARNABA_STACKING_TOPOLOGIES.get(interaction_str)
                    if topology_str:
                        topology = StackingTopology[topology_str]
                        stackings.append(Stacking(nt1, nt2, topology))
                except KeyError:
                    logging.warning(
                        f"Unknown barnaba stacking topology: {interaction_str}"
                    )

    return BaseInteractions.from_structure3d(
        structure3d, base_pairs, stackings, [], [], other_interactions
    )
