#! /usr/bin/env python
import argparse
import csv
import logging
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy
import numpy.typing
import orjson
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
    Residue,
    Saenger,
    Stacking,
    StackingTopology,
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
    Mapping2D3D,
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
    pairs: List[Tuple[Residue3D, Residue3D, int]]
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
    structure: Structure3D, model: int = 1
) -> Tuple[List[BasePair], List[BasePhosphate], List[BaseRibose]]:
    # put all donors and acceptors into a KDTree
    coordinates = []
    coordinates_atom_map: Dict[Tuple[float, float, float], Atom] = {}
    coordinates_type_map: Dict[Tuple[float, float, float], str] = {}
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if residue.model != model:
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


def find_stackings(structure: Structure3D, model: int = 1) -> List[Stacking]:
    # put all nitrogen ring centers into a KDTree
    coordinates = []
    coordinates_residue_map: Dict[Tuple[float, float, float], Residue3D] = {}
    for residue in structure.residues:
        if residue.model != model:
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
    tertiary_structure: Structure3D, model: int = 1
) -> BaseInteractions:
    base_pairs, base_phosphate, base_ribose = find_pairs(tertiary_structure, model)
    stackings = find_stackings(tertiary_structure, model)
    return BaseInteractions(base_pairs, stackings, base_ribose, base_phosphate, [])


def extract_secondary_structure(
    tertiary_structure: Structure3D, model: int = 1, find_gaps: bool = False
) -> BaseInteractions:
    base_interactions = extract_base_interactions(tertiary_structure, model)
    mapping = Mapping2D3D(
        tertiary_structure,
        base_interactions.basePairs,
        base_interactions.stackings,
        find_gaps,
    )
    stems, single_strands, hairpins, loops = mapping.bpseq.elements
    return Structure2D(
        base_interactions,
        str(mapping.bpseq),
        mapping.dot_bracket,
        mapping.extended_dot_bracket,
        stems,
        single_strands,
        hairpins,
        loops,
    )


def write_json(path: str, structure2d: BaseInteractions):
    with open(path, "wb") as f:
        f.write(orjson.dumps(structure2d))


def write_csv(path: str, structure2d: Structure2D):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["nt1", "nt2", "type", "classification-1", "classification-2"])
        for base_pair in structure2d.baseInteractions.basePairs:
            writer.writerow(
                [
                    base_pair.nt1.full_name,
                    base_pair.nt2.full_name,
                    "base pair",
                    base_pair.lw.value,
                    base_pair.saenger.value or ""
                    if base_pair.saenger is not None
                    else "",
                ]
            )
        for stacking in structure2d.baseInteractions.stackings:
            writer.writerow(
                [
                    stacking.nt1.full_name,
                    stacking.nt2.full_name,
                    "stacking",
                    stacking.topology.value if stacking.topology is not None else "",
                    "",
                ]
            )
        for base_phosphate in structure2d.baseInteractions.basePhosphateInteractions:
            writer.writerow(
                [
                    base_phosphate.nt1.full_name,
                    base_phosphate.nt2.full_name,
                    "base-phosphate interaction",
                    base_phosphate.bph.value if base_phosphate.bph is not None else "",
                    "",
                ]
            )
        for base_ribose in structure2d.baseInteractions.basePhosphateInteractions:
            writer.writerow(
                [
                    base_ribose.nt1.full_name,
                    base_ribose.nt2.full_name,
                    "base-ribose interaction",
                    base_ribose.bph.value if base_ribose.bph is not None else "",
                    "",
                ]
            )
        for other in structure2d.baseInteractions.otherInteractions:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument("--bpseq", help="(optional) path to output BPSEQ file")
    parser.add_argument("--csv", help="(optional) path to output CSV file")
    parser.add_argument(
        "--json",
        help="(optional) path to output JSON file",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="(optional) if set, the program will print extended secondary structure to the standard output",
    )
    parser.add_argument(
        "--find-gaps",
        action="store_true",
        help="(optional) if set, the program will detect gaps and break the PDB chain into two or more strands; "
        f"the gap is defined as O3'-P distance greater then {1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT}",
    )
    parser.add_argument("--dot", help="(optional) path to output DOT file")
    args = parser.parse_args()

    file = handle_input_file(args.input)
    structure3d = read_3d_structure(file, 1)
    structure2d = extract_secondary_structure(structure3d, 1, args.find_gaps)

    if args.csv:
        write_csv(args.csv, structure2d)

    if args.json:
        write_json(args.json, structure2d)

    if args.bpseq:
        write_bpseq(args.bpseq, structure2d.bpseq)

    if args.extended:
        print(structure2d.extendedDotBracket)
    else:
        print(structure2d.dotBracket)

    if args.dot:
        print(BpSeq.from_string(structure2d.bpseq).graphviz)


if __name__ == "__main__":
    main()
