#!/usr/bin/env python3
"""
Unifier - A tool to check consistency of residue names across multiple PDB/mmCIF files.

This script accepts a list of PDB or mmCIF files, parses them using the parser_v2 module,
and compares residue names across all structures to check for consistency.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure


def determine_file_format(file_path: str) -> str:
    """
    Determine the format of a structure file based on its extension.

    Parameters:
    -----------
    file_path : str
        Path to the structure file

    Returns:
    --------
    str
        'PDB' for .pdb files, 'mmCIF' for .cif files, or None for unsupported formats
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdb":
        return "PDB"
    elif ext in [".cif", ".mmcif"]:
        return "mmCIF"
    else:
        return None


def parse_structure_file(file_path: str) -> Optional[Structure]:
    """
    Parse a structure file and return a Structure object.

    Parameters:
    -----------
    file_path : str
        Path to the structure file

    Returns:
    --------
    Optional[Structure]
        Structure object if parsing was successful, None otherwise
    """
    file_format = determine_file_format(file_path)
    if not file_format:
        print(f"Error: Unsupported file format for {file_path}", file=sys.stderr)
        return None

    try:
        with open(file_path, "r") as f:
            if file_format == "PDB":
                atoms_df = parse_pdb_atoms(f)
            else:  # mmCIF
                atoms_df = parse_cif_atoms(f)

            if atoms_df.empty:
                print(f"Warning: No atoms found in {file_path}", file=sys.stderr)
                return None

            return Structure(atoms_df)
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}", file=sys.stderr)
        return None


def get_residue_key(residue) -> Tuple[str, int, Optional[str]]:
    """
    Create a unique key for a residue based on chain ID, residue number, and insertion code.

    Parameters:
    -----------
    residue : Residue
        Residue object

    Returns:
    --------
    Tuple[str, int, Optional[str]]
        Tuple containing (chain_id, residue_number, insertion_code)
    """
    return (residue.chain_id, residue.residue_number, residue.insertion_code)


def compare_residue_names(structures: List[Tuple[str, Structure]]) -> Dict:
    """
    Compare residue names across multiple structures.

    Parameters:
    -----------
    structures : List[Tuple[str, Structure]]
        List of tuples containing (file_path, Structure)

    Returns:
    --------
    Dict
        Dictionary with residue keys as keys and a dictionary of file paths and residue names as values
    """
    # Dictionary to store residue names for each position across all structures
    residue_names = defaultdict(dict)

    # Collect residue names from all structures
    for file_path, structure in structures:
        for residue in structure.residues:
            key = get_residue_key(residue)
            residue_names[key][file_path] = residue.residue_name

    return residue_names


def find_inconsistencies(residue_names: Dict) -> Dict:
    """
    Find inconsistencies in residue names across structures.

    Parameters:
    -----------
    residue_names : Dict
        Dictionary with residue keys as keys and a dictionary of file paths and residue names as values

    Returns:
    --------
    Dict
        Dictionary with inconsistent residue keys as keys and a dictionary of file paths and residue names as values
    """
    inconsistencies = {}

    for key, names_by_file in residue_names.items():
        # Get unique residue names for this position
        unique_names = set(names_by_file.values())

        # If there's more than one unique name, this position is inconsistent
        if len(unique_names) > 1:
            inconsistencies[key] = names_by_file

    return inconsistencies


def format_residue_key(key: Tuple[str, int, Optional[str]]) -> str:
    """
    Format a residue key for display.

    Parameters:
    -----------
    key : Tuple[str, int, Optional[str]]
        Tuple containing (chain_id, residue_number, insertion_code)

    Returns:
    --------
    str
        Formatted string representation of the residue
    """
    chain_id, residue_number, insertion_code = key
    if insertion_code:
        return f"{chain_id}:{residue_number}{insertion_code}"
    else:
        return f"{chain_id}:{residue_number}"


def print_inconsistencies(inconsistencies: Dict, file_paths: List[str]) -> None:
    """
    Print inconsistencies in a tabular format.

    Parameters:
    -----------
    inconsistencies : Dict
        Dictionary with inconsistent residue keys as keys and a dictionary of file paths and residue names as values
    file_paths : List[str]
        List of file paths to ensure consistent order in output
    """
    if not inconsistencies:
        print("No inconsistencies found. All residue names match across structures.")
        return

    # Get base file names for display
    base_names = [os.path.basename(path) for path in file_paths]

    # Print header
    header = "Position | " + " | ".join(base_names)
    print(header)
    print("-" * len(header))

    # Sort keys for consistent output
    sorted_keys = sorted(inconsistencies.keys(), key=lambda k: (k[0], k[1], k[2] or ""))

    # Print each inconsistency
    for key in sorted_keys:
        names_by_file = inconsistencies[key]
        position = format_residue_key(key)

        # Get residue name for each file, or "-" if not present
        row_values = [names_by_file.get(path, "-") for path in file_paths]

        print(f"{position:8} | " + " | ".join(f"{val:3}" for val in row_values))


def main():
    """Main function to run the unifier tool."""
    parser = argparse.ArgumentParser(
        description="Check consistency of residue names across multiple PDB/mmCIF files."
    )
    parser.add_argument("files", nargs="+", help="PDB or mmCIF files to compare")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print all residues, not just inconsistencies",
    )
    parser.add_argument(
        "-o", "--output", help="Output file for results (default: stdout)"
    )

    args = parser.parse_args()

    # Parse all structure files
    structures = []
    for file_path in args.files:
        structure = parse_structure_file(file_path)
        if structure:
            structures.append((file_path, structure))

    if len(structures) < 2:
        print(
            "Error: At least two valid structure files are required for comparison",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compare residue names across structures
    residue_names = compare_residue_names(structures)

    # Find inconsistencies
    inconsistencies = find_inconsistencies(residue_names)

    # Redirect output if specified
    original_stdout = sys.stdout
    if args.output:
        try:
            sys.stdout = open(args.output, "w")
        except Exception as e:
            print(f"Error opening output file: {str(e)}", file=sys.stderr)
            sys.exit(1)

    try:
        # Print results
        if args.verbose:
            print_inconsistencies(residue_names, [s[0] for s in structures])
        else:
            print_inconsistencies(inconsistencies, [s[0] for s in structures])
    finally:
        # Restore stdout
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout

    # Print summary to stderr
    total_residues = len(residue_names)
    inconsistent_residues = len(inconsistencies)
    print(
        f"\nSummary: {inconsistent_residues} inconsistencies found out of {total_residues} total residues.",
        file=sys.stderr,
    )

    # Return non-zero exit code if inconsistencies were found
    if inconsistent_residues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
