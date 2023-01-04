import argparse
import csv
import logging
import os
from enum import Enum
from functools import cached_property
from typing import List, Optional

from scipy.spatial import KDTree

from rnapolis.metareader import read_metadata
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Atom, Residue3D

CARBON_RADIUS = 0.6
NITROGEN_RADIUS = 0.54
OXYGEN_RADIUS = 0.53
PHOSPHORUS_RADIUS = 0.94

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


class AtomType(Enum):
    C = "C"
    N = "N"
    O = "O"
    P = "P"

    @cached_property
    def radius(self) -> float:
        if self.value == "C":
            return CARBON_RADIUS
        elif self.value == "N":
            return NITROGEN_RADIUS
        elif self.value == "O":
            return OXYGEN_RADIUS
        elif self.value == "P":
            return PHOSPHORUS_RADIUS
        raise RuntimeError(f"Unknown atom type: {self}")

    def matches(self, atom: Atom):
        return atom.name.strip().startswith(self.value)


def find_clashes(
    residues: List[Residue3D],
    atom_type: AtomType,
    ignore_occupancy: bool,
    ignore_autoclashes: bool,
    nucleic_acid_only: bool,
    require_same_atom_name: bool,
):
    reference_residues = []
    reference_atoms = []
    coordinates = []

    for residue in residues:
        if (
            nucleic_acid_only is True and residue.is_nucleotide
        ) or nucleic_acid_only is False:
            for atom in residue.atoms:
                if atom_type.matches(atom):
                    reference_residues.append(residue)
                    reference_atoms.append(atom)
                    coordinates.append(atom.coordinates)

    if len(coordinates) < 2:
        return []

    kdtree = KDTree(coordinates)

    result = []

    for i, j in kdtree.query_pairs(atom_type.radius * 2.0):
        ai: Atom = reference_atoms[i]
        aj: Atom = reference_atoms[j]
        ri, rj = reference_residues[i], reference_residues[j]

        if ignore_autoclashes is True and ri == rj:
            continue
        if require_same_atom_name is True and ai.name != aj.name:
            continue

        if (
            ignore_occupancy is True
            or (ai.occupancy or 1.0) + (aj.occupancy or 1.0) > 1.0
        ):
            result.append(
                ((ri, ai), (rj, aj), (ai.occupancy or 1.0) + (aj.occupancy or 1.0))
            )

    return result


def classify_clash(atom_i: Atom, atom_j: Atom, occupancy: float) -> Optional[str]:
    if atom_i.name == "O3'" and atom_j.name in (
        "OP1",
        "OP2",
        "OP3",
        "O1P",
        "O2P",
        "O3P",
    ):
        return "O3'"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument(
        "--ignore-occupancy",
        help="By default clashes are not reported if atoms' occupancies are < 1.0, but you can ignore this check with this argument",
        action="store_true",
    )
    parser.add_argument(
        "--nucleic-acid-only",
        help="By default all kind of clashes will be found, but you can focus only on nucleic acids chains",
        action="store_true",
    )
    parser.add_argument(
        "--ignore-autoclashes",
        help="By default clashes will be reported even in scope of the same residue, but you can disable this behaviour",
        action="store_true",
    )
    parser.add_argument(
        "--require-same-atom-name",
        help="By default any two clashing atoms are reported (e.g. P vs OP1), but when this is set only report those clashes of the same atom name (e.g. P vs P)",
        action="store_true",
    )
    parser.add_argument("--csv", help="Store result in CSV format")
    args = parser.parse_args()

    with open(args.input) as f:
        structure3d = read_3d_structure(f, 1)

    clashing_chains = {}
    max_occupancy_residues = {}
    max_occupancy_chains = {}

    for atom_type in AtomType:
        clashes = find_clashes(
            structure3d.residues,
            atom_type,
            args.ignore_occupancy,
            args.ignore_autoclashes,
            args.nucleic_acid_only,
            args.require_same_atom_name,
        )

        if clashes:
            logging.debug(f"Found clashes between atoms of type {atom_type.value}:")

            for pi, pj, occupancy in clashes:
                ri, ai = pi
                rj, aj = pj

                chain_key = (ri.chain, rj.chain)
                residue_key = (ri, rj)
                if chain_key not in clashing_chains:
                    clashing_chains[chain_key] = {}
                if residue_key not in clashing_chains[chain_key]:
                    clashing_chains[chain_key][residue_key] = set()
                clashing_chains[chain_key][residue_key].add((ai, aj, occupancy))

                max_occupancy_residues[(ri, rj)] = max(
                    [max_occupancy_residues.get((ri, rj), 0.0), occupancy]
                )
                max_occupancy_chains[(ri.chain, rj.chain)] = max(
                    [max_occupancy_residues.get((ri.chain, rj.chain), 0.0), occupancy]
                )

    if clashing_chains:
        for ci, cj in sorted(clashing_chains):
            print(
                f"Clashes found between chains {ci} and {cj} with occupancy sum of clashing atoms at maximum {max_occupancy_chains[(ci,cj)]}"
            )
            for ri, rj in clashing_chains[(ci, cj)]:
                print(
                    f"    Clashes found between residues {ri} and {rj} with occupancy sum of clashing atoms at maximum {max_occupancy_residues[(ri,rj)]}"
                )
                for ai, aj, occupancy in sorted(clashing_chains[(ci, cj)][(ri, rj)]):
                    print(
                        f"        Clashes found between atoms {ai.name} and {aj.name} with occupancy sum of {occupancy}"
                    )

        if args.csv:
            metadata = read_metadata(args.input, ["exptl", "refine"])

            with open(args.csv, "w") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Filename",
                        "Experimental method",
                        "Resolution",
                        "Atom 1",
                        "Atom 2",
                        "Occupancy sum",
                        "Classification",
                    ]
                )

                for ci, cj in sorted(clashing_chains):
                    for ri, rj in clashing_chains[(ci, cj)]:
                        for ai, aj, occupancy in sorted(
                            clashing_chains[(ci, cj)][(ri, rj)]
                        ):
                            writer.writerow(
                                [
                                    f"{os.path.splitext(os.path.basename(args.input))[0]}",
                                    metadata["exptl"][0]["method"],
                                    metadata["refine"][0]["ls_d_res_high"],
                                    f"{ri} {ai.name}",
                                    f"{rj} {aj.name}",
                                    occupancy,
                                    classify_clash(ai, aj, occupancy),
                                ]
                            )


if __name__ == "__main__":
    main()
