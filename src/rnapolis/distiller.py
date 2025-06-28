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

    parser.add_argument(
        "--max-subcluster-size",
        type=int,
        default=2,
        help="Maximum subcluster size to merge (default: 2). Find the first merge event where one subcluster exceeds this size.",
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


def find_all_thresholds_and_clusters(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray, file_paths: List[Path]
) -> Tuple[List[dict], Optional[dict]]:
    """
    Find all threshold values where cluster assignments change and generate cluster data.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Square distance matrix
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    file_paths : List[Path]
        List of file paths corresponding to structures

    Returns:
    --------
    Tuple[List[dict], Optional[dict]]
        Tuple of (list of threshold cluster data, max subcluster merge info)
    """
    print("Finding all threshold values where cluster assignments change...")

    # Extract merge distances from linkage matrix (column 2)
    # These are the exact thresholds where cluster assignments change
    merge_distances = linkage_matrix[:, 2]

    # Sort thresholds in ascending order (all thresholds, no range filtering)
    valid_thresholds = np.sort(merge_distances)

    print(f"Testing {len(valid_thresholds)} threshold values where clustering changes:")

    threshold_data = []

    for threshold in valid_thresholds:
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

        print(
            f"  Threshold {threshold:.6f}: {n_clusters} clusters, sizes: {cluster_sizes}"
        )

        # Find medoids for each cluster
        medoids = find_cluster_medoids(list(clusters.values()), distance_matrix)

        # Create threshold data entry
        threshold_entry = {"nrmsd_threshold": float(threshold), "clusters": []}

        for cluster_indices, medoid_idx in zip(clusters.values(), medoids):
            representative = str(file_paths[medoid_idx])
            members = [
                str(file_paths[idx]) for idx in cluster_indices if idx != medoid_idx
            ]

            threshold_entry["clusters"].append(
                {"representative": representative, "members": members}
            )

        threshold_data.append(threshold_entry)

    return threshold_data, None


def find_structure_clusters(
    structures: List[Structure],
    visualize: bool = False,
    rmsd_method: str = "quaternions",
) -> np.ndarray:
    """
    Find clusters of almost identical structures using hierarchical clustering.

    Parameters:
    -----------
    structures : List[Structure]
        List of parsed structures to analyze
    visualize : bool
        Whether to show dendrogram and scatter plot visualization
    rmsd_method : str
        RMSD calculation method ("quaternions" or "svd")

    Returns:
    --------
    np.ndarray
        Distance matrix between all structures
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

    # Return distance matrix for further processing
    return distance_matrix


def find_max_subcluster_merge(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    max_subcluster_size: int,
) -> Optional[dict]:
    """
    Find the first merge event where one subcluster exceeds the maximum size limit.

    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    distance_matrix : np.ndarray
        Square distance matrix
    file_paths : List[Path]
        List of file paths corresponding to structures
    max_subcluster_size : int
        Maximum allowed subcluster size for merging

    Returns:
    --------
    Optional[dict]
        Dictionary with merge information, or None if no such merge is found
    """
    n_structures = len(file_paths)

    # Track cluster sizes at each merge step
    # Initially, each structure is its own cluster of size 1
    cluster_sizes = [1] * n_structures

    # Process merge events in order (from smallest to largest distance)
    for i, merge in enumerate(linkage_matrix):
        left_idx, right_idx, distance, new_cluster_size = merge
        left_idx, right_idx = int(left_idx), int(right_idx)

        # Get sizes of the two clusters being merged
        if left_idx < n_structures:
            left_size = 1  # Original structure
        else:
            left_size = cluster_sizes[left_idx]

        if right_idx < n_structures:
            right_size = 1  # Original structure
        else:
            right_size = cluster_sizes[right_idx]

        # Check if either cluster exceeds the maximum size
        if left_size > max_subcluster_size or right_size > max_subcluster_size:
            # This is the first merge where a subcluster exceeds the limit
            threshold = distance

            # Get cluster assignments at this threshold
            labels = fcluster(linkage_matrix, threshold, criterion="distance")
            n_clusters = len(np.unique(labels))

            # Group structure indices by cluster
            clusters = {}
            for j, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(j)

            cluster_sizes_list = [len(cluster) for cluster in clusters.values()]
            cluster_sizes_list.sort(reverse=True)

            # Find medoids for each cluster
            medoids = find_cluster_medoids(list(clusters.values()), distance_matrix)

            # Create result data
            result = {
                "nrmsd_threshold": float(threshold),
                "merge_info": {
                    "left_subcluster_size": left_size,
                    "right_subcluster_size": right_size,
                    "max_subcluster_size_limit": max_subcluster_size,
                    "merge_step": i + 1,
                },
                "clusters": [],
            }

            for cluster_indices, medoid_idx in zip(clusters.values(), medoids):
                representative = str(file_paths[medoid_idx])
                members = [
                    str(file_paths[idx]) for idx in cluster_indices if idx != medoid_idx
                ]

                result["clusters"].append(
                    {"representative": representative, "members": members}
                )

            return result

        # Update cluster size for the new merged cluster
        new_cluster_idx = n_structures + i
        cluster_sizes.append(left_size + right_size)

    # No merge exceeded the limit
    return None


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

    print(f"Processing {len(valid_files)} files")

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

    # Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix = find_structure_clusters(
        structures, args.visualize, args.rmsd_method
    )

    # Get all threshold data and max subcluster merge info
    linkage_matrix = linkage(squareform(distance_matrix), method="complete")
    all_threshold_data, max_subcluster_merge = find_all_thresholds_and_clusters(
        distance_matrix, linkage_matrix, valid_files
    )

    # Find the first merge event that exceeds max subcluster size
    max_subcluster_merge = find_max_subcluster_merge(
        linkage_matrix, distance_matrix, valid_files, args.max_subcluster_size
    )

    # Show visualizations if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot dendrogram
            dendrogram(
                linkage_matrix,
                labels=[f"Structure {i}" for i in range(len(structures))],
                ax=ax1,
            )
            ax1.set_title("Hierarchical Clustering Dendrogram")
            ax1.set_xlabel("Structure Index")
            ax1.set_ylabel("nRMSD Distance")

            # Plot threshold vs cluster count scatter plot
            thresholds = [entry["nrmsd_threshold"] for entry in all_threshold_data]
            cluster_counts = [len(entry["clusters"]) for entry in all_threshold_data]

            ax2.scatter(thresholds, cluster_counts, alpha=0.7, s=30)
            ax2.set_xlabel("nRMSD Threshold")
            ax2.set_ylabel("Number of Clusters")
            ax2.set_title("Threshold vs Cluster Count")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Always save the plot when --visualize is used
            plt.savefig("clustering_analysis.png", dpi=300, bbox_inches="tight")
            print("Clustering analysis plots saved to clustering_analysis.png")

            # Try to show interactively, but don't fail if it doesn't work
            try:
                plt.show()
            except Exception:
                print("Note: Could not display plot interactively, but saved to file")

        except ImportError:
            print(
                "Warning: matplotlib not available, skipping visualization",
                file=sys.stderr,
            )

    # Print summary
    print(f"\nFound {len(all_threshold_data)} different clustering configurations")
    print(
        f"Threshold range: {all_threshold_data[0]['nrmsd_threshold']:.6f} to {all_threshold_data[-1]['nrmsd_threshold']:.6f}"
    )
    print(
        f"Cluster count range: {len(all_threshold_data[-1]['clusters'])} to {len(all_threshold_data[0]['clusters'])}"
    )

    # Print max subcluster merge information
    if max_subcluster_merge:
        merge_info = max_subcluster_merge["merge_info"]
        print(
            f"\nFirst merge exceeding max subcluster size ({args.max_subcluster_size}):"
        )
        print(f"  Threshold: {max_subcluster_merge['nrmsd_threshold']:.6f}")
        print(f"  Merge step: {merge_info['merge_step']}")
        print(
            f"  Subcluster sizes: {merge_info['left_subcluster_size']} + {merge_info['right_subcluster_size']}"
        )
        print(f"  Resulting clusters: {len(max_subcluster_merge['clusters'])}")
    else:
        print(
            f"\nNo merge events exceeded max subcluster size ({args.max_subcluster_size})"
        )

    # Save comprehensive JSON with all thresholds and max subcluster merge info
    if args.output_json:
        output_data = {
            "all_thresholds": all_threshold_data,
            "max_subcluster_merge": max_subcluster_merge,
            "parameters": {
                "max_subcluster_size": args.max_subcluster_size,
                "rmsd_method": args.rmsd_method,
            },
        }

        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nComprehensive clustering results saved to {args.output_json}")
    else:
        print("\nUse --output-json to save clustering results to a file")


if __name__ == "__main__":
    main()
