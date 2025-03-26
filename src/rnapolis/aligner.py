#!/usr/bin/env python3
import argparse
import os
import tempfile

import pandas as pd

from rnapolis.parser import is_cif
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms, write_cif, write_pdb
from rnapolis.tertiary_v2 import Structure


def main():
    """Main function to run the unifier tool."""
    parser = argparse.ArgumentParser(description="Align two PDB or mmCIF files.")
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument(
        "--format",
        "-f",
        help="Output format (possible values: PDB, mmCIF, keep. Default: keep)",
        default="keep",
    )
    parser.add_argument("pdb1", help="First PDB or mmCIF file")
    parser.add_argument("pdb2", help="Second PDB or mmCIF file")
    args = parser.parse_args()

    from pymol import cmd

    cmd.load(args.pdb1, "pdb1")
    cmd.load(args.pdb2, "pdb2")
    cmd.align("pdb1", "pdb2", object="aligned", cycles=0)

    pdb1_aligned = []
    pdb2_aligned = []

    with tempfile.NamedTemporaryFile("wt+", suffix=".aln") as f:
        cmd.save(f.name, "aligned")
        f.seek(0)

        for line in f:
            if line.startswith("pdb1"):
                pdb1_aligned.append(line.split()[1])
            elif line.startswith("pdb2"):
                pdb2_aligned.append(line.split()[1])

    pdb1_aligned = "".join(pdb1_aligned)
    pdb2_aligned = "".join(pdb2_aligned)
    residues_to_remove = {"pdb1": [], "pdb2": []}

    i, j = 0, 0
    for c1, c2 in zip(pdb1_aligned, pdb2_aligned):
        if c1 == c2 == "-":
            continue  # Should not happen to have gap aligned to gap, but just in case

        if c1 == c2:
            i += 1
            j += 1
            continue

        if c1 == "-":
            residues_to_remove["pdb2"].append(j)
            j += 1
            continue

        if c2 == "-":
            residues_to_remove["pdb1"].append(i)
            i += 1
            continue

        if c1 != c2:
            residues_to_remove["pdb1"].append(i)
            residues_to_remove["pdb2"].append(j)
            i += 1
            j += 1
            continue

        raise ValueError("This should not happen!")

    if not residues_to_remove["pdb1"] and not residues_to_remove["pdb2"]:
        print("Structures are already aligned")

    structures = {}
    for key, path in [("pdb1", args.pdb1), ("pdb2", args.pdb2)]:
        with open(path) as f:
            if is_cif(f):
                atoms = parse_cif_atoms(f)
            else:
                atoms = parse_pdb_atoms(f)

        structures[key] = Structure(atoms).residues

    for key, residues in structures.items():
        for i in sorted(residues_to_remove[key], reverse=True):
            del residues[i]

    # Write output
    os.makedirs(args.output, exist_ok=True)

    for (key, residues), path in zip(structures.items(), [args.pdb1, args.pdb2]):
        base, _ = os.path.splitext(os.path.basename(path))

        if args.format == "keep":
            format = residues[0].atoms.attrs["format"]
        else:
            format = args.format

        ext = ".pdb" if format == "PDB" else ".cif"

        with open(f"{args.output}/{base}{ext}", "w") as f:
            df = pd.concat([residue.atoms for residue in residues])

            if format == "PDB":
                write_pdb(df, f)
            else:
                write_cif(df, f)


if __name__ == "__main__":
    main()
