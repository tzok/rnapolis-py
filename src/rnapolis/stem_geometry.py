"""Experimental stem geometry and PyMOL visualization helpers.

This module is intentionally not imported by the annotator or adapter core.
The inter-stem calculations are experimental and currently untested; they may
be fixed or replaced independently of the secondary structure pipeline.
"""

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy
import numpy.typing
from scipy.stats import vonmises

from rnapolis.common import Stem
from rnapolis.tertiary import Mapping2D3D, Residue3D, calculate_torsion_angle_coords


@dataclass(frozen=True, order=True)
class InterStemParameters:
    """Geometric parameters describing relationships between two stems."""

    stem1_idx: int
    stem2_idx: int
    type: Optional[str]
    torsion: Optional[float]
    min_endpoint_distance: Optional[float]
    torsion_angle_pdf: Optional[float]
    min_endpoint_distance_pdf: Optional[float]
    coaxial_probability: Optional[float]


def _calculate_pair_centroid(
    residue1: Residue3D, residue2: Residue3D
) -> Optional[numpy.typing.NDArray[numpy.floating]]:
    """Calculate the geometric mean of base atoms for a pair of residues."""
    base_atoms = []
    for residue in [residue1, residue2]:
        base_atom_names = Residue3D.nucleobase_heavy_atoms.get(
            residue.one_letter_name.upper(), set()
        )
        if not base_atom_names:
            logging.warning(
                f"Could not find base atom definition for residue {residue.full_name}"
            )
            continue
        for atom in residue.atoms:
            if atom.name in base_atom_names:
                base_atoms.append(atom)

    if not base_atoms:
        logging.warning(
            f"No base atoms found for pair {residue1.full_name} - {residue2.full_name}"
        )
        return None

    coordinates = [atom.coordinates for atom in base_atoms]
    return numpy.mean(coordinates, axis=0)


def get_stem_coordinates(
    mapping: Mapping2D3D, stem: Stem
) -> List[numpy.typing.NDArray[numpy.floating]]:
    """Calculate the geometric centroid for each base pair in a stem."""
    all_pair_centroids = []
    stem_len = stem.strand5p.last - stem.strand5p.first + 1

    for i in range(stem_len):
        idx5p = stem.strand5p.first + i
        idx3p = stem.strand3p.last - i
        try:
            res5p = mapping.bpseq_index_to_residue_map[idx5p]
            res3p = mapping.bpseq_index_to_residue_map[idx3p]
            centroid = _calculate_pair_centroid(res5p, res3p)
            if centroid is not None:
                all_pair_centroids.append(centroid)
        except KeyError:
            logging.warning(
                f"Could not find residues for pair {idx5p}-{idx3p} in stem {stem}"
            )
            continue

    return all_pair_centroids


def calculate_inter_stem_parameters(
    mapping: Mapping2D3D, stem1: Stem, stem2: Stem, kappa: float = 10.0
) -> Optional[Dict[str, Union[str, float]]]:
    """Calculate geometric parameters between two stems."""
    stem1_centroids = get_stem_coordinates(mapping, stem1)
    stem2_centroids = get_stem_coordinates(mapping, stem2)

    if len(stem1_centroids) < 2 or len(stem2_centroids) < 2:
        logging.warning(
            f"Cannot calculate inter-stem parameters for stems {stem1} and {stem2}: "
            f"Insufficient base pairs ({len(stem1_centroids)} and "
            f"{len(stem2_centroids)} respectively)."
        )
        return None

    s1_first, s1_last = stem1_centroids[0], stem1_centroids[-1]
    s2_first, s2_last = stem2_centroids[0], stem2_centroids[-1]

    endpoint_distances = {
        "cs55": numpy.linalg.norm(s1_first - s2_first),
        "cs53": numpy.linalg.norm(s1_first - s2_last),
        "cs35": numpy.linalg.norm(s1_last - s2_first),
        "cs33": numpy.linalg.norm(s1_last - s2_last),
    }

    min_endpoint_distance = min(endpoint_distances.values())
    closest_pair_key = min(endpoint_distances, key=endpoint_distances.get)

    a_rna_twist = 32.7
    mu_degrees = 0.0

    if closest_pair_key == "cs55":
        s1p1, s1p2 = stem1_centroids[1], stem1_centroids[0]
        s2p1, s2p2 = stem2_centroids[0], stem2_centroids[1]
        mu_degrees = 180.0 - a_rna_twist
    elif closest_pair_key == "cs53":
        s1p1, s1p2 = stem1_centroids[1], stem1_centroids[0]
        s2p1, s2p2 = stem2_centroids[-1], stem2_centroids[-2]
        mu_degrees = 0.0 - a_rna_twist
    elif closest_pair_key == "cs35":
        s1p1, s1p2 = stem1_centroids[-2], stem1_centroids[-1]
        s2p1, s2p2 = stem2_centroids[0], stem2_centroids[1]
        mu_degrees = 0.0 + a_rna_twist
    elif closest_pair_key == "cs33":
        s1p1, s1p2 = stem1_centroids[-2], stem1_centroids[-1]
        s2p1, s2p2 = stem2_centroids[-1], stem2_centroids[-2]
        mu_degrees = 180.0 + a_rna_twist
    else:
        logging.error(
            f"Unexpected closest pair key: {closest_pair_key}. "
            "Cannot calculate parameters."
        )
        return None

    torsion_radians = calculate_torsion_angle_coords(s1p1, s1p2, s2p1, s2p2)

    mu_radians = math.radians(mu_degrees)
    vm_dist = vonmises(kappa=kappa, loc=mu_radians)

    torsion_probability = vm_dist.pdf(torsion_radians)
    distance_probability = distance_pdf(min_endpoint_distance)

    max_torsion_probability = vm_dist.pdf(mu_radians)
    max_distance_probability = 1.0
    normalization_factor = max_torsion_probability * max_distance_probability

    coaxial_probability = 0.0
    if normalization_factor > 1e-9:
        probability_product = torsion_probability * distance_probability
        coaxial_probability = probability_product / normalization_factor
        coaxial_probability = max(0.0, min(1.0, coaxial_probability))

    return {
        "type": closest_pair_key,
        "torsion_angle": math.degrees(torsion_radians),
        "min_endpoint_distance": min_endpoint_distance,
        "torsion_angle_pdf": torsion_probability,
        "min_endpoint_distance_pdf": distance_probability,
        "coaxial_probability": coaxial_probability,
    }


def calculate_all_inter_stem_parameters(
    mapping: Mapping2D3D,
) -> List[InterStemParameters]:
    """Calculate parameters for all valid pairs of stems in a mapping."""
    stems = mapping.bpseq.elements[0]
    inter_stem_params = []
    for i, j in itertools.combinations(range(len(stems)), 2):
        stem1 = stems[i]
        stem2 = stems[j]

        if (stem1.strand5p.last - stem1.strand5p.first + 1) > 1 and (
            stem2.strand5p.last - stem2.strand5p.first + 1
        ) > 1:
            params = calculate_inter_stem_parameters(mapping, stem1, stem2)
            if params is not None:
                inter_stem_params.append(
                    InterStemParameters(
                        stem1_idx=i,
                        stem2_idx=j,
                        type=params["type"],
                        torsion=params["torsion_angle"],
                        min_endpoint_distance=params["min_endpoint_distance"],
                        torsion_angle_pdf=params["torsion_angle_pdf"],
                        min_endpoint_distance_pdf=params["min_endpoint_distance_pdf"],
                        coaxial_probability=params["coaxial_probability"],
                    )
                )
    return inter_stem_params


def distance_pdf(
    x: float,
    lower_bound: float = 3.0,
    upper_bound: float = 7.0,
    steepness: float = 5.0,
) -> float:
    """Calculate a plateau probability density for a distance."""
    max_exponent = 700.0

    exponent1 = -steepness * (x - lower_bound)
    exponent1 = min(exponent1, max_exponent)
    sigmoid1 = 1.0 / (1.0 + math.exp(exponent1))

    exponent2 = steepness * (x - upper_bound)
    exponent2 = min(exponent2, max_exponent)
    sigmoid2 = 1.0 / (1.0 + math.exp(exponent2))

    probability = sigmoid1 * sigmoid2
    return max(0.0, min(1.0, probability))


def generate_pymol_script(mapping: Mapping2D3D, stems: List[Stem]) -> str:
    """Generate a PyMOL script visualizing stems as cylinders and dihedrals."""
    pymol_commands = []
    radius = 0.5
    r, g, b = 1.0, 0.0, 0.0

    for stem_idx, stem in enumerate(stems):
        try:
            res5p_first = mapping.bpseq_index_to_residue_map[stem.strand5p.first]
            res5p_last = mapping.bpseq_index_to_residue_map[stem.strand5p.last]
            res3p_first = mapping.bpseq_index_to_residue_map[stem.strand3p.first]
            res3p_last = mapping.bpseq_index_to_residue_map[stem.strand3p.last]

            chain5p = (
                res5p_first.auth.chain if res5p_first.auth else res5p_first.label.chain
            )
            num5p_first = (
                res5p_first.auth.number
                if res5p_first.auth
                else res5p_first.label.number
            )
            num5p_last = (
                res5p_last.auth.number if res5p_last.auth else res5p_last.label.number
            )

            chain3p = (
                res3p_first.auth.chain if res3p_first.auth else res3p_first.label.chain
            )
            num3p_first = (
                res3p_first.auth.number
                if res3p_first.auth
                else res3p_first.label.number
            )
            num3p_last = (
                res3p_last.auth.number if res3p_last.auth else res3p_last.label.number
            )

            selection_str = (
                f"{chain5p}/{num5p_first}-{num5p_last}/ or "
                f"{chain3p}/{num3p_first}-{num3p_last}/"
            )
            pymol_commands.append(f"select stem{stem_idx}, {selection_str}")

        except (KeyError, AttributeError) as e:
            logging.warning(
                f"Could not generate selection string for stem {stem_idx}: "
                f"Missing residue data ({e})"
            )

        centroids = get_stem_coordinates(mapping, stem)

        if len(centroids) < 2:
            continue

        for centroid_idx, centroid in enumerate(centroids):
            x, y, z = centroid
            pseudoatom_name = f"stem{stem_idx}_centroid{centroid_idx}"
            pymol_commands.append(
                f"pseudoatom {pseudoatom_name}, pos=[{x:.3f}, {y:.3f}, {z:.3f}]"
            )

        for seg_idx in range(len(centroids) - 1):
            p1 = centroids[seg_idx]
            p2 = centroids[seg_idx + 1]
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            cgo_object = (
                f"[ 9.0, {x1:.3f}, {y1:.3f}, {z1:.3f}, "
                f"{x2:.3f}, {y2:.3f}, {z2:.3f}, {radius}, "
                f"{r}, {g}, {b}, {r}, {g}, {b} ]"
            )
            pymol_commands.append(
                f'cmd.load_cgo({cgo_object}, "stem_{stem_idx}_seg_{seg_idx}")'
            )

        if len(centroids) >= 4:
            for i in range(len(centroids) - 3):
                pa1 = f"stem{stem_idx}_centroid{i}"
                pa2 = f"stem{stem_idx}_centroid{i + 1}"
                pa3 = f"stem{stem_idx}_centroid{i + 2}"
                pa4 = f"stem{stem_idx}_centroid{i + 3}"
                dihedral_name = f"stem{stem_idx}_dihedral{i}"
                pymol_commands.append(
                    f"dihedral {dihedral_name}, {pa1}, {pa2}, {pa3}, {pa4}"
                )

    return "\n".join(pymol_commands)
