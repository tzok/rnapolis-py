import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure, Residue


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
        "--jobs",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel jobs for nRMSD computation (default: {cpu_count()})",
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


def compute_nrmsd(residues1: List[Residue], residues2: List[Residue]) -> float:
    """
    Compute normalized RMSD between two lists of residues.

    Parameters:
    -----------
    residues1 : List[Residue]
        First list of residues
    residues2 : List[Residue]
        Second list of residues (must have same length as residues1)

    Returns:
    --------
    float
        Normalized RMSD (RMSD / sqrt(number of atom pairs))
    """
    if len(residues1) != len(residues2):
        raise ValueError("Residue lists must have the same length")

    # Define atom sets for different residue types
    backbone_atoms = {
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "C3'",
        "C2'",
        "C1'",
        "O4'",
        "O3'",
        "O2'",
    }
    ribose_atoms = {"C1'", "C2'", "C3'", "C4'", "O4'", "O2'", "O3'"}

    # Purine ring atoms (two rings)
    purine_ring_atoms = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}

    # Pyrimidine ring atoms (one ring)
    pyrimidine_ring_atoms = {"N1", "C2", "N3", "C4", "C5", "C6"}

    # Define purine and pyrimidine residue names
    purines = {"A", "G", "DA", "DG"}
    pyrimidines = {"C", "U", "T", "DC", "DT"}

    atom_pairs = []

    for res1, res2 in zip(residues1, residues2):
        res1_name = res1.residue_name
        res2_name = res2.residue_name

        if res1_name == res2_name:
            # Same residue type - collect all matching atom pairs
            for atom1 in res1.atoms_list:
                atom2 = res2.find_atom(atom1.name)
                if atom2 is not None:
                    atom_pairs.append((atom1.coordinates, atom2.coordinates))

        elif res1_name in purines and res2_name in purines:
            # Both purines - backbone + ribose + purine rings
            target_atoms = backbone_atoms | ribose_atoms | purine_ring_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    atom_pairs.append((atom1.coordinates, atom2.coordinates))

        elif res1_name in pyrimidines and res2_name in pyrimidines:
            # Both pyrimidines - backbone + ribose + pyrimidine ring
            target_atoms = backbone_atoms | ribose_atoms | pyrimidine_ring_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    atom_pairs.append((atom1.coordinates, atom2.coordinates))

        else:
            # Purine-pyrimidine or other combinations - only backbone + ribose
            target_atoms = backbone_atoms | ribose_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    atom_pairs.append((atom1.coordinates, atom2.coordinates))

    if not atom_pairs:
        return float("inf")  # No matching atoms found

    # Convert to numpy arrays
    coords1 = np.array([pair[0] for pair in atom_pairs])
    coords2 = np.array([pair[1] for pair in atom_pairs])

    # Compute optimal superposition using Kabsch algorithm
    rmsd = compute_rmsd_with_superposition(coords1, coords2)

    # Return normalized RMSD
    nrmsd = rmsd / np.sqrt(len(atom_pairs))
    return nrmsd


def compute_rmsd_with_superposition(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute RMSD between two sets of coordinates after optimal superposition.

    Parameters:
    -----------
    coords1 : np.ndarray
        First set of coordinates (N x 3)
    coords2 : np.ndarray
        Second set of coordinates (N x 3)

    Returns:
    --------
    float
        RMSD after optimal superposition
    """
    # Center the coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)

    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2

    # Compute the cross-covariance matrix
    H = centered1.T @ centered2

    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation to centered2
    rotated2 = centered2 @ R.T

    # Compute RMSD
    diff = centered1 - rotated2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd


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
    Find optimal clustering threshold using silhouette analysis.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        Square distance matrix
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering

    Returns:
    --------
    float
        Optimal threshold value
    """
    print("Finding optimal threshold using silhouette analysis...")

    # Candidate distance thresholds
    grid = np.linspace(0.05, 0.30, 60)
    best_score, best_threshold = -1.0, None

    for threshold in grid:
        labels = fcluster(linkage_matrix, threshold, criterion="distance")

        # Silhouette analysis needs at least 2 clusters
        if labels.max() < 2:
            continue

        # Skip if all points are in one cluster
        if len(np.unique(labels)) < 2:
            continue

        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            if score > best_score:
                best_score, best_threshold = score, threshold
        except ValueError:
            # Skip invalid configurations
            continue

    if best_threshold is None:
        # Fallback to a reasonable default if silhouette analysis fails
        print("Warning: Silhouette analysis failed, using default threshold 0.1")
        return 0.1

    print(
        f"Optimal threshold: {best_threshold:.4f} (silhouette score: {best_score:.4f})"
    )
    return best_threshold


def compute_nrmsd_pair(args):
    """Helper function for parallel nRMSD computation."""
    i, j, nucleotides_i, nucleotides_j = args
    nrmsd = compute_nrmsd(nucleotides_i, nucleotides_j)
    return i, j, nrmsd


def find_structure_clusters(
    structures: List[Structure],
    threshold: float,
    visualize: bool = False,
    n_jobs: int = None,
) -> List[List[int]]:
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
    n_jobs : int
        Number of parallel jobs for nRMSD computation

    Returns:
    --------
    List[List[int]]
        List of clusters, where each cluster is a list of structure indices
    """
    n_structures = len(structures)

    if n_structures == 1:
        return [[0]]

    # Get nucleotide residues for each structure
    nucleotide_lists = []
    for structure in structures:
        nucleotides = [
            residue for residue in structure.residues if residue.is_nucleotide
        ]
        nucleotide_lists.append(nucleotides)

    # Compute nRMSD distance matrix in parallel
    print(
        f"Computing pairwise nRMSD distances using {n_jobs or cpu_count()} processes..."
    )
    distance_matrix = np.zeros((n_structures, n_structures))

    # Prepare arguments for parallel computation
    pairs = []
    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            pairs.append((i, j, nucleotide_lists[i], nucleotide_lists[j]))

    # Compute nRMSD values in parallel
    if n_jobs == 1:
        # Sequential computation
        results = [compute_nrmsd_pair(pair) for pair in pairs]
    else:
        # Parallel computation
        with Pool(processes=n_jobs) as pool:
            results = pool.map(compute_nrmsd_pair, pairs)

    # Fill the distance matrix
    for i, j, nrmsd in results:
        distance_matrix[i, j] = nrmsd
        distance_matrix[j, i] = nrmsd
        print(f"  Structure {i} vs {j}: nRMSD = {nrmsd:.4f}")

    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distance_matrix)

    # Perform hierarchical clustering with complete linkage
    linkage_matrix = linkage(condensed_distances, method="complete")

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(distance_matrix, linkage_matrix)

    # Show dendrogram and nRMSD distribution if requested
    if visualize:
        try:
            import matplotlib.pyplot as plt

            # Create subplots for dendrogram and histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot dendrogram
            dendrogram(
                linkage_matrix,
                labels=[f"Structure {i}" for i in range(n_structures)],
                color_threshold=threshold,
                ax=ax1,
            )
            ax1.set_title("Hierarchical Clustering Dendrogram")
            ax1.set_xlabel("Structure Index")
            ax1.set_ylabel("nRMSD Distance")
            ax1.axhline(
                y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
            )
            ax1.legend()

            # Plot nRMSD distribution
            nrmsd_values = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            ax2.hist(nrmsd_values, bins=20, alpha=0.7, edgecolor="black")
            ax2.axvline(
                x=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}"
            )
            ax2.set_title("Distribution of nRMSD Values")
            ax2.set_xlabel("nRMSD")
            ax2.set_ylabel("Frequency")
            ax2.legend()

            plt.tight_layout()

            # Always save the plot when --visualize is used
            plt.savefig("dendrogram.png", dpi=300, bbox_inches="tight")
            print("Dendrogram and nRMSD distribution saved to dendrogram.png")

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
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    # Return as list of lists
    return list(clusters.values())


def main():
    """Main entry point for the distiller CLI tool."""
    args = parse_arguments()

    # Validate input files
    valid_files = validate_input_files(args.files)

    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    threshold_msg = f"auto-selected" if args.threshold is None else f"{args.threshold}"
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
    clusters = find_structure_clusters(
        structures, args.threshold, args.visualize, args.jobs
    )

    # Output results
    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {len(cluster)} structures")
        for structure_idx in cluster:
            print(f"  - {valid_files[structure_idx]}")


if __name__ == "__main__":
    main()
