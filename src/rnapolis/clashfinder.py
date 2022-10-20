import argparse
import logging
import os
from enum import Enum
from typing import List

from scipy.spatial import KDTree

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

    @property
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

        if (
            ignore_occupancy is True
            or (ai.occupancy or 1.0) + (aj.occupancy or 1.0) > 1.0
        ):
            result.append(
                ((ri, ai), (rj, aj), (ai.occupancy or 1.0) + (aj.occupancy or 1.0))
            )

    return result


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
    args = parser.parse_args()

    with open(args.input) as f:
        structure3d = read_3d_structure(f, 1)

    clashing_residues = set()
    clashing_chains = set()
    max_occupancy_residues = dict()
    max_occupancy_chains = dict()

    for atom_type in AtomType:
        clashes = find_clashes(
            structure3d.residues,
            atom_type,
            args.ignore_occupancy,
            args.ignore_autoclashes,
            args.nucleic_acid_only,
        )

        if clashes:
            logging.debug(f"Found clashes between atoms of type {atom_type.value}:")

            for pi, pj, occupancy in clashes:
                ri, ai = pi
                rj, aj = pj
                clashing_residues.add((ri, rj))
                clashing_chains.add((ri.chain, rj.chain))

                max_occupancy_residues[(ri, rj)] = max(
                    [max_occupancy_residues.get((ri, rj), 0.0), occupancy]
                )
                max_occupancy_chains[(ri.chain, rj.chain)] = max(
                    [max_occupancy_residues.get((ri.chain, rj.chain), 0.0), occupancy]
                )

                logging.debug(
                    f"  {ri} atom {ai.name} with {rj} atom {aj.name} (sum of occupancies: {occupancy})"
                )

    if clashing_chains:
        for ci, cj in sorted(clashing_chains):
            print(
                f"Clashes found between chains {ci} and {cj} with occupancy sum of clashing atoms at maximum {max_occupancy_chains[(ci,cj)]}"
            )
        print()
        for ri, rj in sorted(clashing_residues):
            print(
                f"Clashes found between residues {ri} and {rj} with occupancy sum of clashing atoms at maximum {max_occupancy_residues[(ri,rj)]}"
            )


if __name__ == "__main__":
    main()
