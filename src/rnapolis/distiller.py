import argparse
import json
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
from tqdm import tqdm

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure, Residue

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU computation")


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

    parser.add_argument(
        "--output-json",
        type=str,
        help="Output JSON file to save clustering results",
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


def compute_rmsd_batch_gpu(
    coords1_batch: np.ndarray, coords2_batch: np.ndarray, atom_counts: np.ndarray
) -> np.ndarray:
    """
    Compute RMSD for a batch of coordinate pairs using GPU acceleration.

    Parameters:
    -----------
    coords1_batch : np.ndarray
        Batch of first coordinate sets (batch_size, N, 3)
    coords2_batch : np.ndarray
        Batch of second coordinate sets (batch_size, N, 3)
    atom_counts : np.ndarray
        Number of actual atoms for each pair (batch_size,)

    Returns:
    --------
    np.ndarray
        Array of RMSD values for each pair in the batch
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU computation")

    # Transfer to GPU
    coords1_gpu = cp.asarray(coords1_batch)
    coords2_gpu = cp.asarray(coords2_batch)
    atom_counts_gpu = cp.asarray(atom_counts)

    batch_size = coords1_gpu.shape[0]
    max_atoms = coords1_gpu.shape[1]

    # Create masks for actual atoms (not padding)
    atom_indices = cp.arange(max_atoms)[None, :]  # (1, max_atoms)
    masks = atom_indices < atom_counts_gpu[:, None]  # (batch_size, max_atoms)
    masks_3d = masks[:, :, None]  # (batch_size, max_atoms, 1)

    # Apply masks to coordinates (set padded atoms to 0)
    coords1_masked = coords1_gpu * masks_3d
    coords2_masked = coords2_gpu * masks_3d

    # Compute centroids using only actual atoms
    sum_coords1 = cp.sum(coords1_masked, axis=1)  # (batch_size, 3)
    sum_coords2 = cp.sum(coords2_masked, axis=1)  # (batch_size, 3)
    centroid1 = sum_coords1 / atom_counts_gpu[:, None]  # (batch_size, 3)
    centroid2 = sum_coords2 / atom_counts_gpu[:, None]  # (batch_size, 3)

    # Center coordinates
    centered1 = coords1_masked - centroid1[:, None, :]  # (batch_size, max_atoms, 3)
    centered2 = coords2_masked - centroid2[:, None, :]  # (batch_size, max_atoms, 3)

    # Apply masks again after centering (to ensure padded atoms remain 0)
    centered1 = centered1 * masks_3d
    centered2 = centered2 * masks_3d

    # Compute cross-covariance matrices for all pairs
    H = cp.matmul(centered1.transpose(0, 2, 1), centered2)  # (batch_size, 3, 3)

    # SVD for each matrix in the batch
    U, S, Vt = cp.linalg.svd(H)  # Each is (batch_size, 3, 3)

    # Compute rotation matrices
    R = cp.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))  # (batch_size, 3, 3)

    # Ensure proper rotation (det(R) = 1) for each matrix
    det_R = cp.linalg.det(R)  # (batch_size,)
    flip_mask = det_R < 0

    # Flip the last row of Vt for matrices with negative determinant
    Vt_corrected = Vt.copy()
    Vt_corrected[flip_mask, -1, :] *= -1
    R_corrected = cp.matmul(Vt_corrected.transpose(0, 2, 1), U.transpose(0, 2, 1))

    # Apply rotation to centered2
    rotated2 = cp.matmul(
        centered2, R_corrected.transpose(0, 2, 1)
    )  # (batch_size, max_atoms, 3)

    # Compute RMSD for each pair using only actual atoms
    diff = centered1 - rotated2  # (batch_size, max_atoms, 3)
    squared_diff = cp.sum(diff**2, axis=2)  # (batch_size, max_atoms)

    # Apply mask to exclude padded atoms from RMSD calculation
    masked_squared_diff = squared_diff * masks  # (batch_size, max_atoms)
    sum_squared_diff = cp.sum(masked_squared_diff, axis=1)  # (batch_size,)
    rmsd = cp.sqrt(sum_squared_diff / atom_counts_gpu)  # (batch_size,)

    # Transfer back to CPU
    return cp.asnumpy(rmsd)


def compute_rmsd_batch_cpu(
    coords1_batch: np.ndarray, coords2_batch: np.ndarray, atom_counts: np.ndarray
) -> np.ndarray:
    """
    Compute RMSD for a batch of coordinate pairs using CPU.

    Parameters:
    -----------
    coords1_batch : np.ndarray
        Batch of first coordinate sets (batch_size, N, 3)
    coords2_batch : np.ndarray
        Batch of second coordinate sets (batch_size, N, 3)
    atom_counts : np.ndarray
        Number of actual atoms for each pair (batch_size,)

    Returns:
    --------
    np.ndarray
        Array of RMSD values for each pair in the batch
    """
    batch_size = coords1_batch.shape[0]
    rmsd_values = np.zeros(batch_size)

    for i in range(batch_size):
        # Extract only the actual atoms (not padding)
        n_atoms = atom_counts[i]
        coords1 = coords1_batch[i, :n_atoms]
        coords2 = coords2_batch[i, :n_atoms]
        rmsd_values[i] = compute_rmsd_with_superposition(coords1, coords2)

    return rmsd_values


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


def compute_nrmsd_pair(args):
    """Helper function for parallel nRMSD computation."""
    i, j, nucleotides_i, nucleotides_j = args
    nrmsd = compute_nrmsd(nucleotides_i, nucleotides_j)
    return i, j, nrmsd


def extract_atom_coordinates(
    residues1: List[Residue], residues2: List[Residue]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matching atom coordinates from two residue lists.

    Parameters:
    -----------
    residues1 : List[Residue]
        First list of residues
    residues2 : List[Residue]
        Second list of residues

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Coordinate arrays for matching atoms (N, 3) each
    """
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
    purine_ring_atoms = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}
    pyrimidine_ring_atoms = {"N1", "C2", "N3", "C4", "C5", "C6"}

    purines = {"A", "G", "DA", "DG"}
    pyrimidines = {"C", "U", "T", "DC", "DT"}

    coords1_list = []
    coords2_list = []

    for res1, res2 in zip(residues1, residues2):
        res1_name = res1.residue_name
        res2_name = res2.residue_name

        if res1_name == res2_name:
            # Same residue type - collect all matching atom pairs
            for atom1 in res1.atoms_list:
                atom2 = res2.find_atom(atom1.name)
                if atom2 is not None:
                    coords1_list.append(atom1.coordinates)
                    coords2_list.append(atom2.coordinates)
        elif res1_name in purines and res2_name in purines:
            # Both purines
            target_atoms = backbone_atoms | ribose_atoms | purine_ring_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    coords1_list.append(atom1.coordinates)
                    coords2_list.append(atom2.coordinates)
        elif res1_name in pyrimidines and res2_name in pyrimidines:
            # Both pyrimidines
            target_atoms = backbone_atoms | ribose_atoms | pyrimidine_ring_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    coords1_list.append(atom1.coordinates)
                    coords2_list.append(atom2.coordinates)
        else:
            # Purine-pyrimidine or other combinations
            target_atoms = backbone_atoms | ribose_atoms
            for atom_name in target_atoms:
                atom1 = res1.find_atom(atom_name)
                atom2 = res2.find_atom(atom_name)
                if atom1 is not None and atom2 is not None:
                    coords1_list.append(atom1.coordinates)
                    coords2_list.append(atom2.coordinates)

    if not coords1_list:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    return np.array(coords1_list), np.array(coords2_list)


def compute_nrmsd_batch(
    pairs_data: List[Tuple[int, int, List[Residue], List[Residue]]],
) -> List[Tuple[int, int, float]]:
    """
    Compute nRMSD for a batch of structure pairs using GPU acceleration if available.

    Parameters:
    -----------
    pairs_data : List[Tuple[int, int, List[Residue], List[Residue]]]
        List of (i, j, residues_i, residues_j) tuples

    Returns:
    --------
    List[Tuple[int, int, float]]
        List of (i, j, nrmsd) results
    """
    if not pairs_data:
        return []

    # Extract coordinates for all pairs
    coords1_batch = []
    coords2_batch = []
    atom_counts = []
    valid_pairs = []

    for i, j, residues_i, residues_j in pairs_data:
        coords1, coords2 = extract_atom_coordinates(residues_i, residues_j)

        if len(coords1) == 0:
            # No matching atoms - return infinite nRMSD
            valid_pairs.append((i, j, float("inf")))
            continue

        coords1_batch.append(coords1)
        coords2_batch.append(coords2)
        atom_counts.append(len(coords1))
        valid_pairs.append((i, j, None))  # Placeholder for nRMSD

    if not coords1_batch:
        return valid_pairs

    # Pad coordinates to same length for batching
    max_atoms = max(atom_counts)
    coords1_padded = np.zeros((len(coords1_batch), max_atoms, 3))
    coords2_padded = np.zeros((len(coords2_batch), max_atoms, 3))

    for idx, (coords1, coords2, n_atoms) in enumerate(
        zip(coords1_batch, coords2_batch, atom_counts)
    ):
        coords1_padded[idx, :n_atoms] = coords1
        coords2_padded[idx, :n_atoms] = coords2

    # Convert atom_counts to numpy array
    atom_counts_array = np.array(atom_counts)

    # Compute RMSD using GPU or CPU
    if CUPY_AVAILABLE:
        try:
            rmsd_values = compute_rmsd_batch_gpu(
                coords1_padded, coords2_padded, atom_counts_array
            )
        except Exception as e:
            print(f"GPU computation failed, falling back to CPU: {e}")
            rmsd_values = compute_rmsd_batch_cpu(
                coords1_padded, coords2_padded, atom_counts_array
            )
    else:
        rmsd_values = compute_rmsd_batch_cpu(
            coords1_padded, coords2_padded, atom_counts_array
        )

    # Convert to nRMSD and update results
    results = []
    batch_idx = 0

    for i, j, nrmsd_placeholder in valid_pairs:
        if nrmsd_placeholder is None:
            # This was a valid pair that we computed
            rmsd = rmsd_values[batch_idx]
            n_atoms = atom_counts[batch_idx]
            nrmsd = rmsd / np.sqrt(n_atoms)
            results.append((i, j, nrmsd))
            batch_idx += 1
        else:
            # This was an invalid pair (infinite nRMSD)
            results.append((i, j, nrmsd_placeholder))

    return results


def find_structure_clusters(
    structures: List[Structure],
    threshold: float,
    visualize: bool = False,
    n_jobs: int = None,
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

    # Compute nRMSD distance matrix using batched computation
    batch_size = 1000
    use_gpu = (
        CUPY_AVAILABLE and n_jobs != 1
    )  # Use GPU unless explicitly requesting single-threaded

    print(
        f"Computing pairwise nRMSD distances using {'GPU' if use_gpu else 'CPU'} with batch size {batch_size}..."
    )
    distance_matrix = np.zeros((n_structures, n_structures))

    # Prepare all pairs
    all_pairs = []
    for i in range(n_structures):
        for j in range(i + 1, n_structures):
            all_pairs.append((i, j, nucleotide_lists[i], nucleotide_lists[j]))

    # Process pairs in batches with progress bar
    total_pairs = len(all_pairs)
    num_batches = (total_pairs + batch_size - 1) // batch_size

    with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            batch_pairs = all_pairs[batch_start:batch_end]

            if use_gpu:
                # Use GPU batched computation
                results = compute_nrmsd_batch(batch_pairs)
            else:
                # Use CPU computation (potentially parallel)
                if n_jobs == 1:
                    results = [compute_nrmsd_pair(pair) for pair in batch_pairs]
                else:
                    with Pool(processes=n_jobs) as pool:
                        results = pool.map(compute_nrmsd_pair, batch_pairs)

            # Fill the distance matrix
            for i, j, nrmsd in results:
                distance_matrix[i, j] = nrmsd
                distance_matrix[j, i] = nrmsd

            # Update progress bar
            pbar.set_postfix(
                {"pairs": f"{batch_end}/{total_pairs}", "batch_size": len(batch_pairs)}
            )
            pbar.update(1)

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

    # Return clusters and distance matrix
    return list(clusters.values()), distance_matrix


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
    clusters, distance_matrix = find_structure_clusters(
        structures, args.threshold, args.visualize, args.jobs
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
