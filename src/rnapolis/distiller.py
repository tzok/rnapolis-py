import argparse
import sys
from pathlib import Path
from typing import List


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find clusters of almost identical RNA structures from mmCIF or PDB files"
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input mmCIF or PDB files to analyze"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="nRMSD threshold for clustering (default: 0.1)"
    )
    
    return parser.parse_args()


def validate_input_files(files: List[Path]) -> List[Path]:
    """Validate that input files exist and have appropriate extensions."""
    valid_files = []
    valid_extensions = {'.pdb', '.cif', '.mmcif'}
    
    for file_path in files:
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping", file=sys.stderr)
            continue
            
        if file_path.suffix.lower() not in valid_extensions:
            print(f"Warning: File {file_path} does not have a recognized extension (.pdb, .cif, .mmcif), skipping", file=sys.stderr)
            continue
            
        valid_files.append(file_path)
    
    return valid_files


def find_structure_clusters(files: List[Path], threshold: float) -> List[List[Path]]:
    """
    Find clusters of almost identical structures.
    
    Parameters:
    -----------
    files : List[Path]
        List of structure files to analyze
    threshold : float
        nRMSD threshold for clustering
        
    Returns:
    --------
    List[List[Path]]
        List of clusters, where each cluster is a list of similar structure files
    """
    # TODO: Implement clustering algorithm
    # For now, return each file as its own cluster
    return [[file] for file in files]


def main():
    """Main entry point for the distiller CLI tool."""
    args = parse_arguments()
    
    # Validate input files
    valid_files = validate_input_files(args.files)
    
    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(valid_files)} files with nRMSD threshold {args.threshold}")
    
    # Find clusters
    clusters = find_structure_clusters(valid_files, args.threshold)
    
    # Output results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {len(cluster)} structures")
        for file_path in cluster:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()
