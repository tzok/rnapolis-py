#! /usr/bin/env python
import argparse
import copy
import csv
import logging
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy
import numpy.typing
import orjson
import pandas as pd
from ordered_set import OrderedSet
from scipy.spatial import KDTree

from rnapolis.common import (
    BR,
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BPh,
    BpSeq,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    Saenger,
    Stacking,
    StackingTopology,
    Stem,
    Structure2D,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT,
    BASE_ACCEPTORS,
    BASE_ATOMS,
    BASE_DONORS,
    BASE_EDGES,
    PHOSPHATE_ACCEPTORS,
    RIBOSE_ACCEPTORS,
    Atom,
    Mapping2D3D,  # Added import
    Residue3D,
    Structure3D,
    torsion_angle,
)
from rnapolis.util import handle_input_file

HYDROGEN_BOND_MAX_DISTANCE = 4.0
HYDROGEN_BOND_ANGLE_RANGE = (50.0, 130.0)  # 90 degrees is ideal, so allow +- 40 degrees
STACKING_MAX_DISTANCE = 6.0
STACKING_MAX_ANGLE_BETWEEN_NORMALS = 35.0
STACKING_MAX_ANGLE_BETWEEN_VECTOR_AND_NORMAL = 45.0

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


def angle_between_vectors(
    v1: numpy.typing.NDArray[numpy.floating], v2: numpy.typing.NDArray[numpy.floating]
) -> float:
    return math.acos(numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2))


def detect_cis_trans(residue_i: Residue3D, residue_j: Residue3D) -> Optional[str]:
    c1p_i = residue_i.find_atom("C1'")
    c1p_j = residue_j.find_atom("C1'")

    if residue_i.one_letter_name in "AG":
        n9n1_i = residue_i.find_atom("N9")
    else:
        n9n1_i = residue_i.find_atom("N1")

    if residue_j.one_letter_name in "AG":
        n9n1_j = residue_j.find_atom("N9")
    else:
        n9n1_j = residue_j.find_atom("N1")

    if c1p_i is None or c1p_j is None or n9n1_i is None or n9n1_j is None:
        return None

    torsion = math.degrees(torsion_angle(c1p_i, n9n1_i, n9n1_j, c1p_j))
    return "c" if -90.0 < torsion < 90.0 else "t"


def detect_saenger(
    residue_i: Residue3D, residue_j: Residue3D, lw: LeontisWesthof
) -> Optional[Saenger]:
    key = (f"{residue_i.one_letter_name}{residue_j.one_letter_name}", lw.value)
    if key in Saenger.table():
        return Saenger[Saenger.table()[key]]
    return None


def detect_bph_br_classification(
    donor_residue: Residue3D, donor: Atom, acceptor: Atom
) -> Optional[int]:
    # source: Classification and energetics of the base-phosphate interactions in RNA. Craig L. Zirbel, Judit E. Sponer, Jiri Sponer, Jesse Stombaugh and Neocles B. Leontis
    if donor_residue.one_letter_name == "A":
        if donor.name == "C2":
            return 2
        if donor.name == "N6":
            n1 = donor_residue.find_atom("N1")
            c6 = donor_residue.find_atom("C6")
            if n1 is not None and c6 is not None:
                torsion = math.degrees(torsion_angle(n1, c6, donor, acceptor))
                return 6 if -90.0 < torsion < 90.0 else 7
        if donor.name == "C8":
            return 0

    if donor_residue.one_letter_name == "G":
        if donor.name == "N1":
            return 5
        if donor.name == "N2":
            n3 = donor_residue.find_atom("N3")
            c2 = donor_residue.find_atom("C2")
            if n3 is not None and c2 is not None:
                torsion = math.degrees(torsion_angle(n3, c2, donor, acceptor))
                return 1 if -90.0 < torsion < 90.0 else 3
        if donor.name == "C8":
            return 0

    if donor_residue.one_letter_name == "C":
        if donor.name == "N4":
            n3 = donor_residue.find_atom("N3")
            c4 = donor_residue.find_atom("C4")
            if n3 is not None and c4 is not None:
                torsion = math.degrees(torsion_angle(n3, c4, donor, acceptor))
                return 6 if -90.0 < torsion < 90.0 else 7
        if donor.name == "C5":
            return 9
        if donor.name == "C6":
            return 0

    if donor_residue.one_letter_name == "U":
        if donor.name == "N3":
            return 5
        if donor.name == "C5":
            return 9
        if donor.name == "C6":
            return 0

    if donor_residue.one_letter_name == "T":
        if donor.name == "N3":
            return 5
        if donor.name == "C6":
            return 0
        if donor.name == "C7":
            return 9

    return None


def merge_and_clean_bph_br(
    pairs: List[Tuple[Residue3D, Residue3D, int]],
) -> Dict[Tuple[Residue3D, Residue3D], OrderedSet[int]]:
    bph_br_map: Dict[Tuple[Residue3D, Residue3D], OrderedSet[int]] = defaultdict(
        OrderedSet
    )
    for residue_i, residue_j, classification in pairs:
        bph_br_map[(residue_i, residue_j)].add(classification)
    for bphs_brs in bph_br_map.values():
        # 3BPh and 5BPh simultanously means that it is actually 4BPh
        if 3 in bphs_brs and 5 in bphs_brs:
            bphs_brs.remove(3)
            bphs_brs.remove(5)
            bphs_brs.add(4)
        # 7BPh and 9BPh simultanously means that it is actually 8BPh
        if 7 in bphs_brs and 9 in bphs_brs:
            bphs_brs.remove(7)
            bphs_brs.remove(9)
            bphs_brs.add(8)
    for key, bphs_brs in bph_br_map.items():
        if len(bphs_brs) > 1:
            bph_br_map[key] = OrderedSet([bphs_brs[0]])
    return bph_br_map


def find_pairs(
    structure: Structure3D, model: Optional[int] = None
) -> Tuple[List[BasePair], List[BasePhosphate], List[BaseRibose]]:
    # put all donors and acceptors into a KDTree
    coordinates = []
    coordinates_atom_map: Dict[Tuple[float, float, float], Atom] = {}
    coordinates_type_map: Dict[Tuple[float, float, float], str] = {}
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if model is not None and residue.model != model:
            continue
        acceptors = (
            BASE_ACCEPTORS.get(residue.one_letter_name, [])
            + RIBOSE_ACCEPTORS
            + PHOSPHATE_ACCEPTORS
        )
        donors = BASE_DONORS.get(residue.one_letter_name, [])
        for atom_name in acceptors + donors:
            atom = residue.find_atom(atom_name)
            if atom:
                xyz = (atom.x, atom.y, atom.z)
                coordinates.append(xyz)
                coordinates_atom_map[xyz] = atom
                coordinates_type_map[xyz] = (
                    "acceptor" if atom_name in acceptors else "donor"
                )
                coordinates_residue_map[xyz] = residue

    if len(coordinates) < 2:
        return [], [], []

    kdtree = KDTree(coordinates)

    # find all hydrogen bonds
    hydrogen_bonds = []
    base_phosphate_pairs = []
    base_ribose_pairs = []
    used_atoms: Set[Atom] = set()
    for i, j in kdtree.query_pairs(HYDROGEN_BOND_MAX_DISTANCE):
        type_i = coordinates_type_map[coordinates[i]]
        type_j = coordinates_type_map[coordinates[j]]

        # process only acceptor/donor pairs, not acceptor/acceptor or donor/donor
        if type_i == type_j:
            continue

        atom_i = coordinates_atom_map[coordinates[i]]
        atom_j = coordinates_atom_map[coordinates[j]]

        # skip spurious hydrogen bonds in the same residue
        if (
            atom_i.label is not None
            and atom_i.label is not None
            and atom_i.label == atom_j.label
        ):
            continue
        if (
            atom_i.auth is not None
            and atom_i.auth is not None
            and atom_i.auth == atom_j.auth
        ):
            continue

        residue_i = coordinates_residue_map[coordinates[i]]
        residue_j = coordinates_residue_map[coordinates[j]]
        logging.debug(
            f"Checking pair {residue_i.full_name} {atom_i.name} - {residue_j.full_name} {atom_j.name}"
        )

        # check for base-phosphate contacts
        if (
            (atom_i.name in PHOSPHATE_ACCEPTORS or atom_j.name in PHOSPHATE_ACCEPTORS)
            and atom_i not in used_atoms
            and atom_j not in used_atoms
        ):
            logging.debug("Checking base-phosphate interaction")
            if type_i == "donor":
                donor_residue, acceptor_residue = residue_i, residue_j
                donor_atom, acceptor_atom = atom_i, atom_j
            else:
                donor_residue, acceptor_residue = residue_j, residue_i
                donor_atom, acceptor_atom = atom_j, atom_i
            bph = detect_bph_br_classification(donor_residue, donor_atom, acceptor_atom)
            if bph is not None:
                used_atoms.add(atom_i)
                used_atoms.add(atom_j)
                base_phosphate_pairs.append((donor_residue, acceptor_residue, bph))
            continue

        # check for base-ribose contacts
        if (
            (atom_i.name in RIBOSE_ACCEPTORS or atom_j.name in RIBOSE_ACCEPTORS)
            and atom_i not in used_atoms
            and atom_j not in used_atoms
        ):
            logging.debug("Checking base-ribose interaction")
            if type_i == "donor":
                donor_residue, acceptor_residue = residue_i, residue_j
                donor_atom, acceptor_atom = atom_i, atom_j
            else:
                donor_residue, acceptor_residue = residue_j, residue_i
                donor_atom, acceptor_atom = atom_j, atom_i
            br = detect_bph_br_classification(donor_residue, donor_atom, acceptor_atom)
            if br is not None:
                used_atoms.add(atom_i)
                used_atoms.add(atom_j)
                base_ribose_pairs.append((donor_residue, acceptor_residue, br))
            continue

        # check for base-base contacts
        if residue_i.base_normal_vector is None or residue_j.base_normal_vector is None:
            continue

        logging.debug("Checking base-base interaction")
        vector = atom_i.coordinates - atom_j.coordinates
        angle1 = math.degrees(
            angle_between_vectors(residue_i.base_normal_vector, vector)
        )
        angle2 = math.degrees(
            angle_between_vectors(residue_j.base_normal_vector, vector)
        )
        logging.debug(
            f"Angles between normals and hydrogen bond: {angle1:.2f} and {angle2:.2f}"
        )
        if (
            HYDROGEN_BOND_ANGLE_RANGE[0] < angle1 < HYDROGEN_BOND_ANGLE_RANGE[1]
            and HYDROGEN_BOND_ANGLE_RANGE[0] < angle2 < HYDROGEN_BOND_ANGLE_RANGE[1]
        ):
            hydrogen_bonds.append((atom_i, atom_j, residue_i, residue_j))

    # match hydrogen bonds with base edges
    labels = []
    for atom_i, atom_j, residue_i, residue_j in hydrogen_bonds:
        edges_i = BASE_EDGES.get(residue_i.one_letter_name, dict()).get(
            atom_i.name, None
        )
        edges_j = BASE_EDGES.get(residue_j.one_letter_name, dict()).get(
            atom_j.name, None
        )
        if edges_i is None or edges_j is None:
            continue

        # detect cis/trans
        cis_trans = detect_cis_trans(residue_i, residue_j)
        if cis_trans is None:
            continue

        logging.debug(
            f"Matched {residue_i.full_name} with {residue_j.full_name} as {cis_trans} {edges_i} {edges_j}"
        )

        if residue_i < residue_j:
            for edge_i in edges_i:
                for edge_j in edges_j:
                    labels.append((residue_i, residue_j, cis_trans, edge_i, edge_j))
        else:
            for edge_i in edges_i:
                for edge_j in edges_j:
                    labels.append((residue_j, residue_i, cis_trans, edge_j, edge_i))

    # create a list of base pairs
    base_base_pairs = []
    occupied = set()

    counter = Counter(labels)
    for interaction, hydrogen_bond_count in counter.most_common():
        if hydrogen_bond_count < 2:
            continue

        residue_i, residue_j, cis_trans, edge_i, edge_j = interaction

        if (residue_i, edge_i) in occupied:
            continue
        if (residue_j, edge_j) in occupied:
            continue

        occupied.add((residue_i, edge_i))
        occupied.add((residue_j, edge_j))

        lw = LeontisWesthof[f"{cis_trans}{edge_i}{edge_j}"]
        base_base_pairs.append((residue_i, residue_j, lw))

    base_pairs = []
    for residue_i, residue_j, lw in sorted(base_base_pairs):
        base_pairs.append(
            BasePair(
                Residue(residue_i.label, residue_i.auth),
                Residue(residue_j.label, residue_j.auth),
                lw,
                detect_saenger(residue_i, residue_j, lw),
            )
        )

    bph_map = merge_and_clean_bph_br(sorted(base_phosphate_pairs))
    base_phosphates = []
    for pair, bphs in bph_map.items():
        residue_i, residue_j = pair
        for bph in bphs:
            base_phosphates.append(
                BasePhosphate(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    BPh[f"_{bph}"],
                )
            )

    br_map = merge_and_clean_bph_br(sorted(base_ribose_pairs))
    base_riboses = []
    for pair, brs in br_map.items():
        residue_i, residue_j = pair
        for br in brs:
            base_riboses.append(
                BaseRibose(
                    Residue(residue_i.label, residue_i.auth),
                    Residue(residue_j.label, residue_j.auth),
                    BR[f"_{br}"],
                )
            )

    return base_pairs, base_phosphates, base_riboses


def find_stackings(
    structure: Structure3D, model: Optional[int] = None
) -> List[Stacking]:
    # put all nitrogen ring centers into a KDTree
    coordinates = []
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if model is not None and residue.model != model:
            continue
        base_atoms = BASE_ATOMS.get(residue.one_letter_name, [])
        xs, ys, zs = [], [], []
        for atom_name in base_atoms:
            atom = residue.find_atom(atom_name)
            if atom is not None:
                xs.append(atom.x)
                ys.append(atom.y)
                zs.append(atom.z)
        if len(xs) > 0:
            geometric_center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
            coordinates.append(geometric_center)
            coordinates_residue_map[geometric_center] = residue

    if len(coordinates) < 2:
        return []

    kdtree = KDTree(coordinates)

    # find all stacking interaction
    pairs = []
    for i, j in kdtree.query_pairs(STACKING_MAX_DISTANCE):
        residue_i = coordinates_residue_map[coordinates[i]]
        residue_j = coordinates_residue_map[coordinates[j]]

        # check angle between normals
        normal_i = residue_i.base_normal_vector
        normal_j = residue_j.base_normal_vector
        if normal_i is None or normal_j is None:
            continue

        angle = min(
            [
                angle_between_vectors(normal_i, normal_j),
                angle_between_vectors(-normal_i, normal_j),
            ]
        )
        if math.degrees(angle) > STACKING_MAX_ANGLE_BETWEEN_NORMALS:
            continue

        vector = numpy.array([coordinates[i][k] - coordinates[j][k] for k in (0, 1, 2)])
        angle = min(
            angle_between_vectors(vector, normal_i),
            angle_between_vectors(vector, normal_j),
        )
        if math.degrees(angle) > STACKING_MAX_ANGLE_BETWEEN_VECTOR_AND_NORMAL:
            continue

        same_direction = True if numpy.dot(normal_i, normal_j) > 0.0 else False

        if residue_i < residue_j:
            if same_direction:
                pairs.append((residue_i, residue_j, "upward"))
            else:
                pairs.append((residue_i, residue_j, "inward"))
        else:
            if same_direction:
                pairs.append((residue_j, residue_i, "downward"))
            else:
                pairs.append((residue_j, residue_i, "outward"))

    stackings = []
    for residue_i, residue_j, topology in sorted(pairs):
        nt1 = Residue(residue_i.label, residue_i.auth)
        nt2 = Residue(residue_j.label, residue_j.auth)
        stackings.append(Stacking(nt1, nt2, StackingTopology[topology]))

    return stackings


def extract_base_interactions(
    tertiary_structure: Structure3D, model: Optional[int] = None
) -> BaseInteractions:
    base_pairs, base_phosphate, base_ribose = find_pairs(tertiary_structure, model)
    stackings = find_stackings(tertiary_structure, model)
    return BaseInteractions(base_pairs, stackings, base_ribose, base_phosphate, [])


def generate_pymol_script(mapping: Mapping2D3D, stems: List[Stem]) -> str:
    """Generates a PyMOL script to draw stems as cylinders."""
    pymol_commands = []
    radius = 0.5
    r, g, b = 1.0, 0.0, 0.0  # Red color

    for stem_idx, stem in enumerate(stems):
        # Get residues for selection string
        try:
            res5p_first = mapping.bpseq_index_to_residue_map[stem.strand5p.first]
            res5p_last = mapping.bpseq_index_to_residue_map[stem.strand5p.last]
            res3p_first = mapping.bpseq_index_to_residue_map[stem.strand3p.first]
            res3p_last = mapping.bpseq_index_to_residue_map[stem.strand3p.last]

            # Prefer auth chain/number if available
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

            # Format selection string: select stem0, A/1-5/ or A/10-15/
            selection_str = f"{chain5p}/{num5p_first}-{num5p_last}/ or {chain3p}/{num3p_first}-{num3p_last}/"
            pymol_commands.append(f"select stem{stem_idx}, {selection_str}")

        except (KeyError, AttributeError) as e:
            logging.warning(
                f"Could not generate selection string for stem {stem_idx}: Missing residue data ({e})"
            )

        centroids = mapping.get_stem_coordinates(stem)

        # Need at least 2 centroids to draw a segment
        if len(centroids) < 2:
            # Removed warning log for stems with < 2 base pairs
            continue

        # Create pseudoatoms for each centroid
        for centroid_idx, centroid in enumerate(centroids):
            x, y, z = centroid
            pseudoatom_name = f"stem{stem_idx}_centroid{centroid_idx}"
            pymol_commands.append(
                f"pseudoatom {pseudoatom_name}, pos=[{x:.3f}, {y:.3f}, {z:.3f}]"
            )

        # Draw cylinders between consecutive centroids
        for seg_idx in range(len(centroids) - 1):
            p1 = centroids[seg_idx]
            p2 = centroids[seg_idx + 1]
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            # Format: [CYLINDER, x1, y1, z1, x2, y2, z2, radius, r1, g1, b1, r2, g2, b2]
            # Use 9.0 for CYLINDER code
            # Use same color for both ends
            cgo_object = f"[ 9.0, {x1:.3f}, {y1:.3f}, {z1:.3f}, {x2:.3f}, {y2:.3f}, {z2:.3f}, {radius}, {r}, {g}, {b}, {r}, {g}, {b} ]"
            pymol_commands.append(
                f'cmd.load_cgo({cgo_object}, "stem_{stem_idx}_seg_{seg_idx}")'
            )

        # Calculate and display dihedral angles between consecutive centroids
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


def write_json(path: str, structure2d: Structure2D):
    processed = copy.deepcopy(structure2d)
    processed.bpseq_index = {
        k: Residue(v.label, v.auth) for k, v in structure2d.bpseq_index.items()
    }

    with open(path, "wb") as f:
        # Add OPT_SERIALIZE_NUMPY to handle numpy types like float64
        # Add OPT_NON_STR_KEYS to preserve integer keys in dictionaries
        f.write(
            orjson.dumps(
                processed, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
            )
        )


def write_csv(path: str, structure2d: Structure2D):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["nt1", "nt2", "type", "classification-1", "classification-2"])
        for base_pair in structure2d.base_pairs:
            writer.writerow(
                [
                    base_pair.nt1.full_name,
                    base_pair.nt2.full_name,
                    "base pair",
                    base_pair.lw.value,
                    (
                        base_pair.saenger.value or ""
                        if base_pair.saenger is not None
                        else ""
                    ),
                ]
            )
        for stacking in structure2d.stackings:
            writer.writerow(
                [
                    stacking.nt1.full_name,
                    stacking.nt2.full_name,
                    "stacking",
                    stacking.topology.value if stacking.topology is not None else "",
                    "",
                ]
            )
        for base_phosphate in structure2d.base_phosphate_interactions:
            writer.writerow(
                [
                    base_phosphate.nt1.full_name,
                    base_phosphate.nt2.full_name,
                    "base-phosphate interaction",
                    base_phosphate.bph.value if base_phosphate.bph is not None else "",
                    "",
                ]
            )
        for base_ribose in structure2d.base_ribose_interactions:
            writer.writerow(
                [
                    base_ribose.nt1.full_name,
                    base_ribose.nt2.full_name,
                    "base-ribose interaction",
                    base_ribose.br.value if base_ribose.br is not None else "",
                    "",
                ]
            )
        for other in structure2d.other_interactions:
            writer.writerow(
                [
                    other.nt1.full_name,
                    other.nt2.full_name,
                    "other interaction",
                    "",
                    "",
                ]
            )


def write_bpseq(path: str, bpseq: BpSeq):
    with open(path, "w") as f:
        f.write(str(bpseq))


def add_common_output_arguments(parser: argparse.ArgumentParser):
    """Adds common output and processing arguments to the parser."""
    parser.add_argument("-b", "--bpseq", help="(optional) path to output BPSEQ file")
    parser.add_argument("-c", "--csv", help="(optional) path to output CSV file")
    parser.add_argument(
        "-j",
        "--json",
        help="(optional) path to output JSON file",
    )
    parser.add_argument(
        "-e",
        "--extended",
        action="store_true",
        help="(optional) if set, the program will print extended secondary structure to the standard output",
    )
    parser.add_argument("-d", "--dot", help="(optional) path to output DOT file")
    parser.add_argument(
        "-p", "--pml", help="(optional) path to output PyMOL PML script for stems"
    )
    parser.add_argument(
        "--inter-stem-csv",
        help="(optional) path to output CSV file for inter-stem parameters",
    )
    parser.add_argument(
        "--stems-csv",
        help="(optional) path to output CSV file for stem details",
    )


def unify_structure_data(structure2d: Structure2D, mapping: Mapping2D3D) -> Structure2D:
    """
    Unify structure data by:
    1. Adding missing Saenger classifications to base pairs
    2. Filling in empty residue labels from Structure3D
    """
    # Create a mapping from residue to residue3d for label filling
    residue_to_residue3d = {}
    for residue3d in mapping.structure3d.residues:
        residue_key = Residue(residue3d.label, residue3d.auth)
        residue_to_residue3d[residue_key] = residue3d

    def fill_residue_label(residue: Residue) -> Residue:
        """Fill empty label from Structure3D if available."""
        if residue.label is not None:
            return residue

        # Try to find matching residue3d by auth
        for residue3d in mapping.structure3d.residues:
            if residue.auth == residue3d.auth:
                return Residue(residue3d.label, residue.auth)

        return residue

    # Process base pairs
    unified_base_pairs = []
    for base_pair in structure2d.base_pairs:
        # Fill in missing labels
        nt1 = fill_residue_label(base_pair.nt1)
        nt2 = fill_residue_label(base_pair.nt2)

        # Detect missing Saenger classification
        saenger = base_pair.saenger
        if saenger is None:
            # Find corresponding 3D residues for Saenger detection
            residue3d_1 = residue_to_residue3d.get(Residue(nt1.label, nt1.auth))
            residue3d_2 = residue_to_residue3d.get(Residue(nt2.label, nt2.auth))

            if residue3d_1 is not None and residue3d_2 is not None:
                saenger = detect_saenger(residue3d_1, residue3d_2, base_pair.lw)

        unified_base_pairs.append(BasePair(nt1, nt2, base_pair.lw, saenger))

    # Process other interaction types (fill labels only)
    unified_stackings = []
    for stacking in structure2d.stackings:
        nt1 = fill_residue_label(stacking.nt1)
        nt2 = fill_residue_label(stacking.nt2)
        unified_stackings.append(Stacking(nt1, nt2, stacking.topology))

    unified_base_ribose = []
    for base_ribose in structure2d.base_ribose_interactions:
        nt1 = fill_residue_label(base_ribose.nt1)
        nt2 = fill_residue_label(base_ribose.nt2)
        unified_base_ribose.append(BaseRibose(nt1, nt2, base_ribose.br))

    unified_base_phosphate = []
    for base_phosphate in structure2d.base_phosphate_interactions:
        nt1 = fill_residue_label(base_phosphate.nt1)
        nt2 = fill_residue_label(base_phosphate.nt2)
        unified_base_phosphate.append(BasePhosphate(nt1, nt2, base_phosphate.bph))

    unified_other = []
    for other in structure2d.other_interactions:
        nt1 = fill_residue_label(other.nt1)
        nt2 = fill_residue_label(other.nt2)
        unified_other.append(OtherInteraction(nt1, nt2))

    # Create new Structure2D with unified data
    unified_base_interactions = BaseInteractions(
        unified_base_pairs,
        unified_stackings,
        unified_base_ribose,
        unified_base_phosphate,
        unified_other,
    )

    # Recreate Structure2D with unified interactions
    unified_structure2d, _ = mapping.structure3d.extract_secondary_structure(
        unified_base_interactions, False
    )

    return unified_structure2d


def handle_output_arguments(
    args: argparse.Namespace,
    structure2d: Structure2D,
    mapping: Mapping2D3D,
    input_filename: str,
):
    """Handles writing output based on provided arguments."""
    # Unify the structure data before processing outputs
    unified_structure2d = unify_structure_data(structure2d, mapping)

    input_basename = os.path.basename(input_filename)
    if args.csv:
        write_csv(args.csv, unified_structure2d)

    if args.json:
        write_json(args.json, unified_structure2d)

    if args.bpseq:
        write_bpseq(args.bpseq, unified_structure2d.bpseq)

    if args.extended:
        print(unified_structure2d.extended_dot_bracket)
    else:
        print(unified_structure2d.dot_bracket)

    if args.dot:
        print(BpSeq.from_string(unified_structure2d.bpseq).graphviz)

    if args.pml:
        pml_script = generate_pymol_script(mapping, unified_structure2d.stems)
        with open(args.pml, "w") as f:
            f.write(pml_script)

    if args.inter_stem_csv:
        if unified_structure2d.inter_stem_parameters:
            # Convert list of dataclasses to list of dicts
            params_list = [
                {
                    "stem1_idx": p.stem1_idx,
                    "stem2_idx": p.stem2_idx,
                    "type": p.type,
                    "torsion": p.torsion,
                    "min_endpoint_distance": p.min_endpoint_distance,
                    "torsion_angle_pdf": p.torsion_angle_pdf,
                    "min_endpoint_distance_pdf": p.min_endpoint_distance_pdf,
                    "coaxial_probability": p.coaxial_probability,
                }
                for p in unified_structure2d.interStemParameters
            ]
            df = pd.DataFrame(params_list)
            df["input_basename"] = input_basename
            # Reorder columns to put input_basename first
            cols = ["input_basename"] + [
                col for col in df.columns if col != "input_basename"
            ]
            df = df[cols]
            df.to_csv(args.inter_stem_csv, index=False)
        else:
            logging.warning(
                f"No inter-stem parameters calculated for {input_basename}, CSV file '{args.inter_stem_csv}' will be empty or not created."
            )
            # Optionally create an empty file with headers
            # pd.DataFrame(columns=['input_basename', 'stem1_idx', ...]).to_csv(args.inter_stem_csv, index=False)

    if args.stems_csv:
        if unified_structure2d.stems:
            stems_data = []
            for i, stem in enumerate(unified_structure2d.stems):
                try:
                    res5p_first = mapping.bpseq_index_to_residue_map.get(
                        stem.strand5p.first
                    )
                    res5p_last = mapping.bpseq_index_to_residue_map.get(
                        stem.strand5p.last
                    )
                    res3p_first = mapping.bpseq_index_to_residue_map.get(
                        stem.strand3p.first
                    )
                    res3p_last = mapping.bpseq_index_to_residue_map.get(
                        stem.strand3p.last
                    )

                    stems_data.append(
                        {
                            "stem_idx": i,
                            "strand5p_first_nt_id": res5p_first.full_name
                            if res5p_first
                            else None,
                            "strand5p_last_nt_id": res5p_last.full_name
                            if res5p_last
                            else None,
                            "strand3p_first_nt_id": res3p_first.full_name
                            if res3p_first
                            else None,
                            "strand3p_last_nt_id": res3p_last.full_name
                            if res3p_last
                            else None,
                            "strand5p_sequence": stem.strand5p.sequence,
                            "strand3p_sequence": stem.strand3p.sequence,
                        }
                    )
                except KeyError as e:
                    logging.warning(
                        f"Could not find residue for stem {i} (index {e}), skipping stem details."
                    )
                    continue

            if stems_data:
                df_stems = pd.DataFrame(stems_data)
                df_stems["input_basename"] = input_basename
                # Reorder columns
                stem_cols = ["input_basename", "stem_idx"] + [
                    col
                    for col in df_stems.columns
                    if col not in ["input_basename", "stem_idx"]
                ]
                df_stems = df_stems[stem_cols]
                df_stems.to_csv(args.stems_csv, index=False)
            else:
                logging.warning(
                    f"No valid stem data generated for {input_basename}, CSV file '{args.stems_csv}' will be empty or not created."
                )
        else:
            logging.warning(
                f"No stems found for {input_basename}, CSV file '{args.stems_csv}' will be empty or not created."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument(
        "-f",
        "--find-gaps",
        action="store_true",
        help="(optional) if set, the program will detect gaps and break the PDB chain into two or more strands; "
        f"the gap is defined as O3'-P distance greater then {1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT}",
    )
    add_common_output_arguments(parser)
    args = parser.parse_args()

    file = handle_input_file(args.input)
    structure3d = read_3d_structure(file, None)
    base_interactions = extract_base_interactions(structure3d)
    structure2d, mapping = structure3d.extract_secondary_structure(
        base_interactions, args.find_gaps
    )

    handle_output_arguments(args, structure2d, mapping, args.input)


if __name__ == "__main__":
    main()
