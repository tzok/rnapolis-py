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


def get_atom_names(residue) -> Set[str]:
    """
    Get the set of atom names in a residue.

    Parameters:
    -----------
    residue : Residue
        Residue object

    Returns:
    --------
    Set[str]
        Set of atom names in the residue
    """
    return {atom.name for atom in residue.atoms_list}


def compare_atom_names(structures: List[Tuple[str, Structure]]) -> Dict:
    """
    Compare atom names in each residue across multiple structures.

    Parameters:
    -----------
    structures : List[Tuple[str, Structure]]
        List of tuples containing (file_path, Structure)

    Returns:
    --------
    Dict
        Dictionary with residue keys as keys and a dictionary of file paths and atom name sets as values
    """
    # Dictionary to store atom names for each residue position across all structures
    atom_names = defaultdict(dict)

    # Map of residue keys to residue objects for later reference
    residue_objects = defaultdict(dict)

    # Collect atom names from all structures
    for file_path, structure in structures:
        for residue in structure.residues:
            key = get_residue_key(residue)
            atom_names[key][file_path] = get_atom_names(residue)
            residue_objects[key][file_path] = residue

    return atom_names, residue_objects


def find_atom_inconsistencies(atom_names: Dict) -> Dict:
    """
    Find inconsistencies in atom names across structures.

    Parameters:
    -----------
    atom_names : Dict
        Dictionary with residue keys as keys and a dictionary of file paths and atom name sets as values

    Returns:
    --------
    Dict
        Dictionary with inconsistent residue keys as keys and a dictionary of file paths and atom name sets as values
    """
    inconsistencies = {}

    for key, atoms_by_file in atom_names.items():
        # Get all atom sets
        all_atom_sets = list(atoms_by_file.values())

        # If there are at least two structures to compare
        if len(all_atom_sets) >= 2:
            # Check if all atom sets are the same
            reference_set = all_atom_sets[0]
            for atom_set in all_atom_sets[1:]:
                if atom_set != reference_set:
                    inconsistencies[key] = atoms_by_file
                    break

    return inconsistencies


def print_atom_inconsistencies(
    inconsistencies: Dict, residue_objects: Dict, file_paths: List[str]
) -> None:
    """
    Print atom name inconsistencies in a tabular format.

    Parameters:
    -----------
    inconsistencies : Dict
        Dictionary with inconsistent residue keys as keys and a dictionary of file paths and atom name sets as values
    residue_objects : Dict
        Dictionary with residue keys as keys and a dictionary of file paths and residue objects as values
    file_paths : List[str]
        List of file paths to ensure consistent order in output
    """
    if not inconsistencies:
        print(
            "No atom name inconsistencies found. All residues have the same atoms across structures."
        )
        return

    # Get base file names for display
    base_names = [os.path.basename(path) for path in file_paths]

    # Print header
    print("\nAtom Name Inconsistencies:")
    print("=========================")

    # Sort keys for consistent output
    sorted_keys = sorted(inconsistencies.keys(), key=lambda k: (k[0], k[1], k[2] or ""))

    for key in sorted_keys:
        atoms_by_file = inconsistencies[key]
        position = format_residue_key(key)

        # Get residue name for this position
        residue_name = (
            residue_objects[key][file_paths[0]].residue_name
            if key in residue_objects and file_paths[0] in residue_objects[key]
            else "Unknown"
        )

        print(f"\nResidue {position} ({residue_name}):")

        # Find all unique atoms across all files
        all_atoms = set()
        for atom_set in atoms_by_file.values():
            all_atoms.update(atom_set)

        # Sort atoms for consistent display
        sorted_atoms = sorted(all_atoms)

        # Print header row with file names
        print(f"{'Atom':10} | " + " | ".join(f"{name:15}" for name in base_names))
        print("-" * (12 + 19 * len(base_names)))

        # Print each atom and whether it's present in each file
        for atom in sorted_atoms:
            row = [atom]
            for path in file_paths:
                if path in atoms_by_file and atom in atoms_by_file[path]:
                    row.append("✓")
                else:
                    row.append("✗")

            print(f"{row[0]:10} | " + " | ".join(f"{val:^15}" for val in row[1:]))


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
        "-a",
        "--atoms",
        action="store_true",
        help="Check atom names when residue names are consistent",
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

    # Find inconsistencies in residue names
    residue_inconsistencies = find_inconsistencies(residue_names)

    # Redirect output if specified
    original_stdout = sys.stdout
    if args.output:
        try:
            sys.stdout = open(args.output, "w")
        except Exception as e:
            print(f"Error opening output file: {str(e)}", file=sys.stderr)
            sys.exit(1)

    try:
        # Print residue name results
        if args.verbose:
            print_inconsistencies(residue_names, [s[0] for s in structures])
        else:
            print_inconsistencies(residue_inconsistencies, [s[0] for s in structures])

        # Check atom names if requested and no residue name inconsistencies found
        if args.atoms and not residue_inconsistencies:
            # Compare atom names across structures
            atom_names, residue_objects = compare_atom_names(structures)

            # Find inconsistencies in atom names
            atom_inconsistencies = find_atom_inconsistencies(atom_names)

            # Print atom name inconsistencies
            print_atom_inconsistencies(
                atom_inconsistencies, residue_objects, [s[0] for s in structures]
            )

            # Update exit code if atom inconsistencies were found
            if atom_inconsistencies:
                residue_inconsistencies = True  # This will cause a non-zero exit code
    finally:
        # Restore stdout
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout

    # Print summary to stderr
    total_residues = len(residue_names)
    inconsistent_residues = len(residue_inconsistencies)
    print(
        f"\nSummary: {inconsistent_residues} residue name inconsistencies found out of {total_residues} total residues.",
        file=sys.stderr,
    )

    # Return non-zero exit code if inconsistencies were found
    if residue_inconsistencies:
        sys.exit(1)


if __name__ == "__main__":
    main()
