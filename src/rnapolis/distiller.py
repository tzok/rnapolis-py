import argparse
import sys
from pathlib import Path
from typing import List

from .parser_v2 import parse_pdb_atoms, parse_cif_atoms
from .tertiary_v2 import Structure


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find clusters of almost identical RNA structures from mmCIF or PDB files"
    )

    parser.add_argument(
        "files", nargs="+", type=Path, help="Input mmCIF or PDB files to analyze"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="nRMSD threshold for clustering (default: 0.1)",
    )

    return parser.parse_args()


def validate_input_files(files: List[Path]) -> List[Path]:
    """Validate that input files exist and have appropriate extensions."""
    valid_files = []
    valid_extensions = {".pdb", ".cif", ".mmcif"}

    for file_path in files:
        if not file_path.exists():
            print(
                f"Warning: File {file_path} does not exist, skipping", file=sys.stderr
            )
            continue

        if file_path.suffix.lower() not in valid_extensions:
            print(
                f"Warning: File {file_path} does not have a recognized extension (.pdb, .cif, .mmcif), skipping",
                file=sys.stderr,
            )
            continue

        valid_files.append(file_path)

    return valid_files


def parse_structure_file(file_path: Path) -> Structure:
    """
    Parse a structure file (PDB or mmCIF) into a Structure object.

    Parameters:
    -----------
    file_path : Path
        Path to the structure file

    Returns:
    --------
    Structure
        Parsed structure object
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Determine file type and parse accordingly
        if file_path.suffix.lower() == ".pdb":
            atoms_df = parse_pdb_atoms(content)
        else:  # .cif or .mmcif
            atoms_df = parse_cif_atoms(content)

        return Structure(atoms_df)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        raise


def validate_nucleotide_counts(
    structures: List[Structure], file_paths: List[Path]
) -> None:
    """
    Validate that all structures have the same number of nucleotides.

    Parameters:
    -----------
    structures : List[Structure]
        List of parsed structures
    file_paths : List[Path]
        Corresponding file paths for error reporting

    Raises:
    -------
    SystemExit
        If structures have different numbers of nucleotides
    """
    nucleotide_counts = []

    for structure, file_path in zip(structures, file_paths):
        nucleotide_residues = [
            residue for residue in structure.residues if residue.is_nucleotide
        ]
        nucleotide_counts.append((len(nucleotide_residues), file_path))

    if not nucleotide_counts:
        print("Error: No structures with nucleotides found", file=sys.stderr)
        sys.exit(1)

    # Check if all counts are the same
    first_count = nucleotide_counts[0][0]
    mismatched = [
        (count, path) for count, path in nucleotide_counts if count != first_count
    ]

    if mismatched:
        print(
            "Error: Structures have different numbers of nucleotides:", file=sys.stderr
        )
        print(
            f"Expected: {first_count} nucleotides (from {nucleotide_counts[0][1]})",
            file=sys.stderr,
        )
        for count, path in mismatched:
            print(f"Found: {count} nucleotides in {path}", file=sys.stderr)
        sys.exit(1)

    print(f"All structures have {first_count} nucleotides")


def find_structure_clusters(
    structures: List[Structure], threshold: float
) -> List[List[int]]:
    """
    Find clusters of almost identical structures.

    Parameters:
    -----------
    structures : List[Structure]
        List of parsed structures to analyze
    threshold : float
        nRMSD threshold for clustering

    Returns:
    --------
    List[List[int]]
        List of clusters, where each cluster is a list of structure indices
    """
    # TODO: Implement clustering algorithm
    # For now, return each structure as its own cluster
    return [[i] for i in range(len(structures))]


def main():
    """Main entry point for the distiller CLI tool."""
    args = parse_arguments()

    # Validate input files
    valid_files = validate_input_files(args.files)

    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(valid_files)} files with nRMSD threshold {args.threshold}")

    # Parse all structure files
    print("Parsing structure files...")
    structures = []
    for file_path in valid_files:
        try:
            structure = parse_structure_file(file_path)
            structures.append(structure)
            print(f"  Parsed {file_path}")
        except Exception:
            print(f"  Failed to parse {file_path}, skipping", file=sys.stderr)
            continue

    if not structures:
        print("Error: No structures could be parsed", file=sys.stderr)
        sys.exit(1)

    # Update valid_files to match successfully parsed structures
    valid_files = valid_files[: len(structures)]

    # Validate nucleotide counts
    print("\nValidating nucleotide counts...")
    validate_nucleotide_counts(structures, valid_files)

    # Find clusters
    print("\nFinding structure clusters...")
    clusters = find_structure_clusters(structures, args.threshold)

    # Output results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {len(cluster)} structures")
        for structure_idx in cluster:
            print(f"  - {valid_files[structure_idx]}")


if __name__ == "__main__":
    main()
