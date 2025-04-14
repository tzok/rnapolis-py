#!/usr/bin/env python3
import argparse
import os
import sys
from collections import Counter

import pandas as pd

from rnapolis.parser import is_cif
from rnapolis.parser_v2 import (
    fit_to_pdb,
    parse_cif_atoms,
    parse_pdb_atoms,
    write_cif,
    write_pdb,
)
from rnapolis.tertiary_v2 import Structure


def load_components():
    result = {}
    for residue in "ACGU":
        component = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), f"component_{residue}.csv"
        )
        result[residue] = pd.read_csv(component)
    return result


def main():
    """Main function to run the unifier tool."""
    parser = argparse.ArgumentParser(
        description="Unify content of a set of PDB or mmCIF files."
    )
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument(
        "--format",
        "-f",
        help="Output format (possible values: PDB, mmCIF, keep. Default: keep)",
        default="keep",
    )
    parser.add_argument("files", nargs="+", help="PDB or mmCIF files to compare")
    args = parser.parse_args()

    components = load_components()
    structures = []

    for path in args.files:
        with open(path) as f:
            if is_cif(f):
                atoms = parse_cif_atoms(f)
            else:
                atoms = parse_pdb_atoms(f)

        residues = []

        for residue in Structure(atoms).residues:
            if residue.residue_name not in "ACGU":
                continue

            component = components[residue.residue_name]
            mapping_dict = dict(
                [row["alt_atom_id"], row["atom_id"]] for _, row in component.iterrows()
            )
            valid_names = component["atom_id"]
            valid_names = valid_names[~valid_names.str.startswith("H")]
            valid_order = {value: idx for idx, value in enumerate(valid_names.tolist())}
            column = "name" if residue.format == "PDB" else "auth_atom_id"

            # Replace alternative name with standard name
            residue.atoms[column] = residue.atoms[column].replace(mapping_dict)
            # Leave only standard, non-hydrogen atoms
            residue.atoms = residue.atoms[residue.atoms[column].isin(valid_names)]
            # Reorder atoms
            residue.atoms = residue.atoms.sort_values(
                by=[column], key=lambda col: col.map(valid_order)
            )
            residues.append(residue)

        structures.append((path, residues))

    residues_to_remove = set()
    for path, residues in structures:
        ref_path, ref_residues = structures[0]

        # Validity check 1: residue count must be equal
        if len(residues) != len(ref_residues):
            print(
                f"Number of residues in {path} does not match {ref_path}, cannot continue"
            )
            sys.exit(1)

        # Validity check 2: residue names must be equal
        for i, (residue, ref_residue) in enumerate(zip(residues, ref_residues)):
            if residue.residue_name != ref_residue.residue_name:
                print(
                    f"Residue {str(residue)} in {path} does not match {str(ref_residue)} in {ref_path}, cannot continue"
                )
                sys.exit(1)

        # Find residues with different number of atoms
        for i, (residue, ref_residue) in enumerate(zip(residues, ref_residues)):
            if len(residue.atoms) != len(ref_residue.atoms):
                print(
                    f"Number of atoms in {str(residue)} in {path} does not match {str(ref_residue)} in {ref_path}, will unify this"
                )
                residues_to_remove.add(i)

    # Remove residues with different number of atoms
    for _, residues in structures:
        for i in sorted(residues_to_remove, reverse=True):
            del residues[i]

    # Find most common residue identifiers for each residue
    n = len(structures[0][1])
    counters = [Counter() for _ in range(n)]
    for _, residues in structures:
        for i, residue in enumerate(residues):
            counters[i].update(
                [(residue.chain_id, residue.residue_number, residue.insertion_code)]
            )

    # If any residue has different identifiers, use the most common one in all structures
    for i, counter in enumerate(counters):
        (chain_id, residue_number, insertion_code), count = counter.most_common(1)[0]
        if count != len(structures):
            print(
                f"Residue {i + 1} has different identifiers in different structures, will unify this"
            )
            for _, residues in structures:
                residue = residues[i]
                residue.chain_id = chain_id
                residue.residue_number = residue_number
                residue.insertion_code = insertion_code

    # Write output
    os.makedirs(args.output, exist_ok=True)

    for path, residues in structures:
        base, _ = os.path.splitext(os.path.basename(path))

        if args.format == "keep":
            format = residues[0].atoms.attrs["format"]
        else:
            format = args.format

        ext = ".pdb" if format == "PDB" else ".cif"

        df = pd.concat([residue.atoms for residue in residues])

        try:
            if format == "PDB":
                df_to_write = fit_to_pdb(df)
                with open(f"{args.output}/{base}{ext}", "w") as f:
                    write_pdb(df_to_write, f)
            else:
                with open(f"{args.output}/{base}{ext}", "w") as f:
                    write_cif(df, f)
        except ValueError as e:
            print(
                f"Error processing {path} for PDB output: {e}. Skipping file.",
                file=sys.stderr,
            )
            continue


if __name__ == "__main__":
    main()
