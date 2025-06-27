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
from rnapolis.tertiary_v2 import Residue, Structure

# Define atom sets for different residue types
BACKBONE_RIBOSE_ATOMS = {
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
}
PURINE_CORE_ATOMS = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}
PYRIMIDINE_CORE_ATOMS = {"N1", "C2", "N3", "C4", "C5", "C6"}
ATOMS_A = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS | {"N6"}
ATOMS_G = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS | {"O6"}
ATOMS_C = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"N4", "O2"}
ATOMS_U = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"O4", "O2"}
PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}
RESIDUE_ATOMS_MAP = {
    "A": ATOMS_A,
    "G": ATOMS_G,
    "C": ATOMS_C,
    "U": ATOMS_U,
}


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


def nrmsd_quaternions(residues1, residues2):
    """
    Calculates nRMSD using the Quaternion method.
    residues1 and residues2 are lists of Residue objects.
    """
    # Get paired coordinates
    P, Q = find_paired_coordinates(residues1, residues2)

    # 1. Center coordinates using vectorized operations
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 2. Covariance matrix using matrix multiplication
    C = P_centered.T @ Q_centered

    # 3. K matrix
    K = np.zeros((4, 4))
    K[0, 0] = C[0, 0] + C[1, 1] + C[2, 2]
    K[0, 1] = K[1, 0] = C[1, 2] - C[2, 1]
    K[0, 2] = K[2, 0] = C[2, 0] - C[0, 2]
    K[0, 3] = K[3, 0] = C[0, 1] - C[1, 0]
    K[1, 1] = C[0, 0] - C[1, 1] - C[2, 2]
    K[1, 2] = K[2, 1] = C[0, 1] + C[1, 0]
    K[1, 3] = K[3, 1] = C[0, 2] + C[2, 0]
    K[2, 2] = -C[0, 0] + C[1, 1] - C[2, 2]
    K[2, 3] = K[3, 2] = C[1, 2] + C[2, 1]
    K[3, 3] = -C[0, 0] - C[1, 1] + C[2, 2]

    # 4. Eigenvalue/vector
    eigenvalues, _ = np.linalg.eigh(K)

    # E0 = sum of squared distances from centroids
    E0 = np.sum(P_centered**2) + np.sum(Q_centered**2)

    # The min RMSD squared is (E0 - 2*max_eigenvalue) / N
    N = P.shape[0]
    rmsd_sq = (E0 - 2 * np.max(eigenvalues)) / N

    # Handle potential floating point inaccuracies
    return np.sqrt(max(0.0, rmsd_sq) / N)


def nrmsd_svd(residues1, residues2):
    """
    Calculates nRMSD using SVD decomposition (Kabsch algorithm).
    residues1 and residues2 are lists of Residue objects.
    """
    # Get paired coordinates
    P, Q = find_paired_coordinates(residues1, residues2)

    # 1. Center coordinates
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 2. Compute cross-covariance matrix
    H = P_centered.T @ Q_centered

    # 3. SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # 4. Compute optimal rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 5. Apply rotation to P_centered
    P_rotated = P_centered @ R.T

    # 6. Calculate RMSD
    diff = P_rotated - Q_centered
    rmsd_sq = np.sum(diff**2) / P.shape[0]

    return np.sqrt(rmsd_sq) / np.sqrt(P.shape[0])


def nrmsd_qcp(residues1, residues2):
    """
    Calculates nRMSD using the QCP (Quaternion Characteristic Polynomial) method.
    This implementation follows the BioJava QCP algorithm.
    residues1 and residues2 are lists of Residue objects.
    """
    # Get paired coordinates
    P, Q = find_paired_coordinates(residues1, residues2)

    # 1. Center coordinates
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # 2. Calculate inner product matrix elements
    N = P.shape[0]

    # Calculate cross-covariance matrix elements
    Sxx = np.sum(P_centered[:, 0] * Q_centered[:, 0])
    Sxy = np.sum(P_centered[:, 0] * Q_centered[:, 1])
    Sxz = np.sum(P_centered[:, 0] * Q_centered[:, 2])
    Syx = np.sum(P_centered[:, 1] * Q_centered[:, 0])
    Syy = np.sum(P_centered[:, 1] * Q_centered[:, 1])
    Syz = np.sum(P_centered[:, 1] * Q_centered[:, 2])
    Szx = np.sum(P_centered[:, 2] * Q_centered[:, 0])
    Szy = np.sum(P_centered[:, 2] * Q_centered[:, 1])
    Szz = np.sum(P_centered[:, 2] * Q_centered[:, 2])

    # 3. Calculate E0 (sum of squared distances from centroids)
    E0 = np.sum(P_centered**2) + np.sum(Q_centered**2)

    # 4. Calculate coefficients for the characteristic polynomial (following BioJava)
    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Szz2 = Szz * Szz
    Sxy2 = Sxy * Sxy
    Syz2 = Syz * Syz
    Sxz2 = Sxz * Sxz
    Syx2 = Syx * Syx
    Szy2 = Szy * Szy
    Szx2 = Szx * Szx

    SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz)
    Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2

    c2 = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2)
    c1 = 8.0 * (
        Sxx * Syz * Szy
        + Syy * Szx * Sxz
        + Szz * Sxy * Syx
        - Sxx * Syy * Szz
        - Syz * Szx * Sxy
        - Szy * Syx * Sxz
    )

    SxzpSzx = Sxz + Szx
    SyzpSzy = Syz + Szy
    SxypSyx = Sxy + Syx
    SyzmSzy = Syz - Szy
    SxzmSzx = Sxz - Szx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy

    Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2

    c0 = (
        Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2
        + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2)
        * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2)
        + (-(SxzpSzx) * (SyzmSzy) + (SxymSyx) * (SxxmSyy - Szz))
        * (-(SxzmSzx) * (SyzpSzy) + (SxymSyx) * (SxxmSyy + Szz))
        + (-(SxzpSzx) * (SyzpSzy) - (SxypSyx) * (SxxpSyy - Szz))
        * (-(SxzmSzx) * (SyzmSzy) - (SxypSyx) * (SxxpSyy + Szz))
        + (+(SxypSyx) * (SyzpSzy) + (SxzpSzx) * (SxxmSyy + Szz))
        * (-(SxymSyx) * (SyzmSzy) + (SxzpSzx) * (SxxpSyy + Szz))
        + (+(SxypSyx) * (SyzmSzy) + (SxzmSzx) * (SxxmSyy - Szz))
        * (-(SxymSyx) * (SyzpSzy) + (SxzmSzx) * (SxxpSyy - Szz))
    )

    # 5. Find the largest eigenvalue using Newton-Raphson method
    mxEigenV = E0

    eval_prec = 1e-11
    for i in range(50):
        oldg = mxEigenV
        x2 = mxEigenV * mxEigenV
        b = (x2 + c2) * mxEigenV
        a = b + c1
        delta = (a * mxEigenV + c0) / (2.0 * x2 * mxEigenV + b + a)
        mxEigenV -= delta

        if abs(mxEigenV - oldg) < abs(eval_prec * mxEigenV):
            break

    # 6. Calculate RMSD
    rmsd_sq = 2.0 * (E0 - mxEigenV) / N
    rmsd = np.sqrt(abs(rmsd_sq))

    return rmsd


def nrmsd_validate(residues1, residues2):
    """
    Validates that all RMSD methods produce the same result.
    Uses quaternions method as the primary result after validation.

    Parameters:
    -----------
    residues1 : List[Residue]
        List of residues from the first structure
    residues2 : List[Residue]
        List of residues from the second structure

    Returns:
    --------
    float
        nRMSD value (from quaternions method after validation)

    Raises:
    -------
    ValueError
        If any methods produce significantly different results
    """
    # Calculate using all methods
    result_quaternions = nrmsd_quaternions(residues1, residues2)
    result_svd = nrmsd_svd(residues1, residues2)
    result_qcp = nrmsd_qcp(residues1, residues2)

    # Check if results are approximately equal (within 1e-6 tolerance)
    tolerance = 1e-6

    # Check quaternions vs SVD
    if abs(result_quaternions - result_svd) > tolerance:
        raise ValueError(
            f"RMSD methods disagree: quaternions={result_quaternions:.8f}, "
            f"svd={result_svd:.8f}, difference={abs(result_quaternions - result_svd):.8f}"
        )

    # Check quaternions vs QCP
    if abs(result_quaternions - result_qcp) > tolerance:
        raise ValueError(
            f"RMSD methods disagree: quaternions={result_quaternions:.8f}, "
            f"qcp={result_qcp:.8f}, difference={abs(result_quaternions - result_qcp):.8f}"
        )

    # Check SVD vs QCP
    if abs(result_svd - result_qcp) > tolerance:
        raise ValueError(
            f"RMSD methods disagree: svd={result_svd:.8f}, "
            f"qcp={result_qcp:.8f}, difference={abs(result_svd - result_qcp):.8f}"
        )

    # Return quaternions result as the validated value
    return result_quaternions


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


def find_paired_coordinates(
    residues1: List[Residue], residues2: List[Residue]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find paired coordinates of matching atoms between two residues.

    Parameters:
    -----------
    residues1 : List[Residue]
        List of residues from the first structure
    residues2 : List[Residue]
        List of residues from the second structure

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Tuple of two numpy arrays containing coordinates of matching atom pairs
    """
    all_paired_dfs = []

    for residue1, residue2 in zip(residues1, residues2):
        res_name1 = residue1.residue_name
        res_name2 = residue2.residue_name

        atoms_to_match = None

        if res_name1 == res_name2:
            atoms_to_match = RESIDUE_ATOMS_MAP.get(res_name1)
        elif res_name1 in PURINES and res_name2 in PURINES:
            atoms_to_match = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS
        elif res_name1 in PYRIMIDINES and res_name2 in PYRIMIDINES:
            atoms_to_match = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS
        else:
            atoms_to_match = BACKBONE_RIBOSE_ATOMS

        if residue1.format == "mmCIF":
            df1 = residue1.atoms
        else:
            df1 = parse_cif_atoms(write_cif(residue1.atoms_df))

        if residue2.format == "mmCIF":
            df2 = residue2.atoms
        else:
            df2 = parse_cif_atoms(write_cif(residue2.atoms_df))

        df1_filtered = df1[df1["auth_atom_id"].isin(atoms_to_match)]
        df2_filtered = df2[df2["auth_atom_id"].isin(atoms_to_match)]

        paired_df = pd.merge(
            df1_filtered[["auth_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]],
            df2_filtered[["auth_atom_id", "Cartn_x", "Cartn_y", "Cartn_z"]],
            on="auth_atom_id",
            suffixes=("_1", "_2"),
        )

        if not paired_df.empty:
            all_paired_dfs.append(paired_df)

    final_df = pd.concat(all_paired_dfs, ignore_index=True)
    coords_1 = final_df[["Cartn_x_1", "Cartn_y_1", "Cartn_z_1"]].to_numpy()
    coords_2 = final_df[["Cartn_x_2", "Cartn_y_2", "Cartn_z_2"]].to_numpy()
    return coords_1, coords_2


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

    # Select RMSD function based on method
    if rmsd_method == "quaternions":
        rmsd_func = nrmsd_quaternions
        print("Computing pairwise nRMSD distances using quaternion method...")
    elif rmsd_method == "svd":
        rmsd_func = nrmsd_svd
        print("Computing pairwise nRMSD distances using SVD method...")
    elif rmsd_method == "qcp":
        rmsd_func = nrmsd_qcp
        print("Computing pairwise nRMSD distances using QCP method...")
    elif rmsd_method == "validate":
        rmsd_func = nrmsd_validate
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
            executor.submit(rmsd_func, nucleotides_i, nucleotides_j): (i, j)
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
