import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms, write_cif
from rnapolis.tertiary_v2 import (
    Residue,
    Structure,
    nrmsd_quaternions_residues,
    nrmsd_svd_residues,
    nrmsd_qcp_residues,
    nrmsd_validate_residues,
)


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
        default=None,
        help="nRMSD threshold for clustering (default: auto-select using silhouette analysis)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show dendrogram visualization of clustering",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        help="Output JSON file to save clustering results",
    )

    parser.add_argument(
        "--rmsd-method",
        type=str,
        choices=["quaternions", "svd", "qcp", "validate"],
        default="quaternions",
        help="RMSD calculation method (default: quaternions). Use 'validate' to check all methods agree.",
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


def find_optimal_threshold(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray
) -> float:
    """
    Find all threshold values where cluster assignments change and display them.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Square distance matrix
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering

    Returns:
    --------
    float
        Default threshold value (0.1) for backward compatibility
    """
    print("Finding all threshold values where cluster assignments change...")
    
    # Extract merge distances from linkage matrix (column 2)
    # These are the exact thresholds where cluster assignments change
    merge_distances = linkage_matrix[:, 2]
    
    # Filter to the range we're interested in (0.05 to 0.3)
    valid_thresholds = merge_distances[(merge_distances >= 0.05) & (merge_distances <= 0.3)]
    
    # Sort thresholds in ascending order
    valid_thresholds = np.sort(valid_thresholds)
    
    # Add boundary values if they're not already included
    boundary_thresholds = []
    if len(valid_thresholds) == 0 or valid_thresholds[0] > 0.05:
        boundary_thresholds.append(0.05)
    boundary_thresholds.extend(valid_thresholds)
    if len(valid_thresholds) == 0 or valid_thresholds[-1] < 0.3:
        boundary_thresholds.append(0.3)
    
    thresholds_to_test = np.array(boundary_thresholds)
    
    print(f"Testing {len(thresholds_to_test)} threshold values where clustering changes:")
    
    for threshold in thresholds_to_test:
        labels = fcluster(linkage_matrix, threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        
        # Group structure indices by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        cluster_sizes.sort(reverse=True)  # Sort by size, largest first
        
        print(f"  Threshold {threshold:.4f}: {n_clusters} clusters, sizes: {cluster_sizes}")
    
    # Return a reasonable default threshold for backward compatibility
    # Choose the middle value from our range
    if len(valid_thresholds) > 0:
        default_threshold = valid_thresholds[len(valid_thresholds) // 2] if len(valid_thresholds) > 1 else valid_thresholds[0]
    else:
        default_threshold = 0.1
    
    print(f"\nUsing default threshold: {default_threshold:.4f}")
    return default_threshold


def find_structure_clusters(
    structures: List[Structure],
    threshold: Optional[float] = None,
    visualize: bool = False,
    rmsd_method: str = "quaternions",
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Find clusters of almost identical structures using hierarchical clustering.

    Parameters:
    -----------
    structures : List[Structure]
        List of parsed structures to analyze
    threshold : float
        nRMSD threshold for clustering
    visualize : bool
        Whether to show dendrogram visualization
    rmsd_method : str
        RMSD calculation method ("quaternions" or "svd")

    Returns:
    --------
    List[List[int]]
        List of clusters, where each cluster is a list of structure indices
    """
    n_structures = len(structures)

    if n_structures == 1:
        return [[0]], np.zeros((1, 1))

    # Get nucleotide residues for each structure
    nucleotide_lists = []
    for structure in structures:
        nucleotide_lists.append(
            [residue for residue in structure.residues if residue.is_nucleotide]
        )

    # Select nRMSD function based on method
    if rmsd_method == "quaternions":
        nrmsd_func = nrmsd_quaternions_residues
        print("Computing pairwise nRMSD distances using quaternion method...")
    elif rmsd_method == "svd":
        nrmsd_func = nrmsd_svd_residues
        print("Computing pairwise nRMSD distances using SVD method...")
    elif rmsd_method == "qcp":
        nrmsd_func = nrmsd_qcp_residues
        print("Computing pairwise nRMSD distances using QCP method...")
    elif rmsd_method == "validate":
        nrmsd_func = nrmsd_validate_residues
        print(
            "Computing pairwise nRMSD distances using validation mode (all methods)..."
        )
    else:
        raise ValueError(f"Unknown RMSD method: {rmsd_method}")

    distance_matrix = np.zeros((n_structures, n_structures))

    # Prepare all pairs
    all_pairs = [
        (i, j, nucleotide_lists[i], nucleotide_lists[j])
        for i, j in itertools.combinations(range(n_structures), 2)
    ]

    # Process pairs with progress bar
    with ProcessPoolExecutor() as executor:
        futures_dict = {
            executor.submit(nrmsd_func, nucleotides_i, nucleotides_j): (i, j)
            for i, j, nucleotides_i, nucleotides_j in all_pairs
        }
        results = []
        for future in tqdm(
            futures_dict,
            total=len(futures_dict),
            desc="Computing nRMSD",
            unit="pair",
        ):
            i, j = futures_dict[future]
            nrmsd_value = future.result()
            results.append((i, j, nrmsd_value))

    # Fill the distance matrix
    for i, j, nrmsd in results:
        distance_matrix[i, j] = nrmsd
        distance_matrix[j, i] = nrmsd

    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distance_matrix)

    # Perform hierarchical clustering with complete linkage
    linkage_matrix = linkage(condensed_distances, method="complete")

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(distance_matrix, linkage_matrix)

    # Show dendrogram if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt

            # Create figure for dendrogram only
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Plot dendrogram
            dendrogram(
                linkage_matrix,
                labels=[f"Structure {i}" for i in range(n_structures)],
                color_threshold=threshold,
                ax=ax,
            )
            ax.set_title("Hierarchical Clustering Dendrogram")
            ax.set_xlabel("Structure Index")
            ax.set_ylabel("nRMSD Distance")
            ax.axhline(
                y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
            )
            ax.legend()

            plt.tight_layout()

            # Always save the plot when --visualize is used
            plt.savefig("dendrogram.png", dpi=300, bbox_inches="tight")
            print("Dendrogram saved to dendrogram.png")

            # Try to show interactively, but don't fail if it doesn't work
            try:
                plt.show()
            except Exception:
                print("Note: Could not display plot interactively, but saved to file")

        except ImportError:
            print(
                "Warning: matplotlib not available, skipping dendrogram visualization",
                file=sys.stderr,
            )

    # Get cluster labels using the threshold
    cluster_labels = fcluster(linkage_matrix, threshold, criterion="distance")

    # Group structure indices by cluster
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    # Return clusters and distance matrix
    return list(clusters.values()), distance_matrix


def find_cluster_medoids(
    clusters: List[List[int]], distance_matrix: np.ndarray
) -> List[int]:
    """
    Find the medoid (representative) for each cluster.

    Parameters:
    -----------
    clusters : List[List[int]]
        List of clusters, where each cluster is a list of structure indices
    distance_matrix : np.ndarray
        Square distance matrix between all structures

    Returns:
    --------
    List[int]
        List of medoid indices, one for each cluster
    """
    medoids = []

    for cluster in clusters:
        if len(cluster) == 1:
            # Single element cluster - it's its own medoid
            medoids.append(cluster[0])
        else:
            # Find the element with minimum sum of distances to all other elements in cluster
            min_sum_distance = float("inf")
            medoid = cluster[0]

            for candidate in cluster:
                sum_distance = sum(
                    distance_matrix[candidate, other]
                    for other in cluster
                    if other != candidate
                )
                if sum_distance < min_sum_distance:
                    min_sum_distance = sum_distance
                    medoid = candidate

            medoids.append(medoid)

    return medoids


def main():
    """Main entry point for the distiller CLI tool."""
    args = parse_arguments()

    # Validate input files
    valid_files = validate_input_files(args.files)

    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    threshold_msg = "auto-selected" if args.threshold is None else f"{args.threshold}"
    print(f"Processing {len(valid_files)} files with nRMSD threshold {threshold_msg}")

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
    clusters, distance_matrix = find_structure_clusters(
        structures, args.threshold, args.visualize, args.rmsd_method
    )

    # Find medoids for each cluster
    medoids = find_cluster_medoids(clusters, distance_matrix)

    # Output results
    print(f"\nFound {len(clusters)} clusters:")
    for i, (cluster, medoid_idx) in enumerate(zip(clusters, medoids), 1):
        print(f"Cluster {i}: {len(cluster)} structures")
        print(f"  Representative (medoid): {valid_files[medoid_idx]}")
        for structure_idx in cluster:
            if structure_idx != medoid_idx:
                print(f"  - {valid_files[structure_idx]}")

    # Save to JSON if requested
    if args.output_json:
        json_data = {"clusters": []}

        for i, (cluster, medoid_idx) in enumerate(zip(clusters, medoids), 1):
            cluster_data = {
                "cluster_id": i,
                "representative": str(valid_files[medoid_idx]),
                "members": [
                    str(valid_files[idx]) for idx in cluster if idx != medoid_idx
                ],
            }
            json_data["clusters"].append(cluster_data)

        with open(args.output_json, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\nClustering results saved to {args.output_json}")


if __name__ == "__main__":
    main()
