import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from tqdm import tqdm

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import (
    Structure,
    calculate_torsion_angle,
    nrmsd_qcp_residues,
    nrmsd_quaternions_residues,
    nrmsd_svd_residues,
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
        help="Show dendrogram visualization of clustering (exact mode only)",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        help="Output JSON file to save clustering results (available in both modes)",
    )

    parser.add_argument(
        "--rmsd-method",
        type=str,
        choices=["quaternions", "svd", "qcp", "validate"],
        default="quaternions",
        help="RMSD calculation method (default: quaternions). Use 'validate' to check all methods agree. (exact mode only)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="nRMSD threshold for clustering (default: auto-detect from exponential decay inflection point) (exact mode only)",
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default="nrmsd_cache.json",
        help="Cache file for storing computed nRMSD values (exact mode only, default: nrmsd_cache.json)",
    )

    parser.add_argument(
        "--cache-save-interval",
        type=int,
        default=100,
        help="Save cache to disk every N computations (exact mode only, default: 100)",
    )

    parser.add_argument(
        "--mode",
        choices=["exact", "approximate"],
        default="exact",
        help="Clustering mode switch: --mode exact (default) performs rigorous nRMSD clustering, "
        "--mode approximate performs faster feature-based PCA + FAISS clustering",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Radius in PCA-reduced space for redundancy detection (approximate mode only, default: 10.0)",
    )

    return parser.parse_args()


class NRMSDCache:
    """Cache for storing computed nRMSD values with file metadata."""

    def __init__(self, cache_file: str, save_interval: int = 100):
        self.cache_file = cache_file
        self.save_interval = save_interval
        self.cache: Dict[str, float] = {}
        self.computation_count = 0
        self.load_cache()

    def _get_file_key(self, file_path: Path) -> str:
        """Generate a unique key for a file based on path and modification time."""
        stat = file_path.stat()
        return f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"

    def _get_pair_key(self, file1: Path, file2: Path, rmsd_method: str) -> str:
        """Generate a unique key for a file pair and method."""
        key1 = self._get_file_key(file1)
        key2 = self._get_file_key(file2)
        # Ensure consistent ordering
        if key1 > key2:
            key1, key2 = key2, key1
        combined = f"{key1}|{key2}|{rmsd_method}"
        # Use hash to keep keys manageable
        return hashlib.md5(combined.encode()).hexdigest()

    def load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
                print(
                    f"Loaded {len(self.cache)} cached nRMSD values from {self.cache_file}"
                )
            except Exception as e:
                print(
                    f"Warning: Could not load cache file {self.cache_file}: {e}",
                    file=sys.stderr,
                )
                self.cache = {}
        else:
            print(f"No existing cache file found at {self.cache_file}")

    def save_cache(self, silent: bool = False):
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
            if not silent:
                print(f"Saved {len(self.cache)} cached values to {self.cache_file}")
        except Exception as e:
            print(
                f"Warning: Could not save cache file {self.cache_file}: {e}",
                file=sys.stderr,
            )

    def get(self, file1: Path, file2: Path, rmsd_method: str) -> Optional[float]:
        """Get cached nRMSD value if available."""
        key = self._get_pair_key(file1, file2, rmsd_method)
        return self.cache.get(key)

    def set(self, file1: Path, file2: Path, rmsd_method: str, value: float):
        """Store nRMSD value in cache."""
        key = self._get_pair_key(file1, file2, rmsd_method)
        self.cache[key] = value
        self.computation_count += 1

        # Save periodically (silently to avoid disrupting progress bar)
        if self.computation_count % self.save_interval == 0:
            self.save_cache(silent=True)


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


# ----------------------------------------------------------------------


def run_exact(structures: List[Structure], valid_files: List[Path], args) -> None:
    """
    Exact mode: nRMSD-based clustering workflow (previously in main).
    Produces the same outputs/visualisations as before.
    """
    # Initialize cache
    print("\nInitializing nRMSD cache...")
    cache = NRMSDCache(args.cache_file, args.cache_save_interval)

    # Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix = find_structure_clusters(
        structures, valid_files, cache, args.visualize, args.rmsd_method
    )

    # Build linkage matrix
    linkage_matrix = linkage(squareform(distance_matrix), method="complete")

    # Determine threshold
    if args.threshold is None:
        optimal_threshold = determine_optimal_threshold(distance_matrix, linkage_matrix)
    else:
        optimal_threshold = args.threshold
        print(f"Using user-specified threshold: {optimal_threshold}")

    # Collect threshold data
    all_threshold_data = find_all_thresholds_and_clusters(
        distance_matrix, linkage_matrix, valid_files
    )
    threshold_clustering = get_clustering_at_threshold(
        linkage_matrix, distance_matrix, valid_files, optimal_threshold
    )

    # Visualisation (re-uses the earlier logic)
    if args.visualize:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            dendrogram(
                linkage_matrix,
                labels=[f"Structure {i}" for i in range(len(structures))],
                ax=ax1,
                color_threshold=optimal_threshold,
            )
            ax1.axhline(
                y=optimal_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold = {optimal_threshold:.6f}",
            )
            ax1.set_title("Hierarchical Clustering Dendrogram")
            ax1.set_xlabel("Structure Index")
            ax1.set_ylabel("nRMSD Distance")
            ax1.legend()

            thresholds = np.array(
                [entry["nrmsd_threshold"] for entry in all_threshold_data]
            )
            cluster_counts = np.array(
                [len(entry["clusters"]) for entry in all_threshold_data]
            )

            x_smooth, y_smooth, inflection_x = fit_exponential_decay(
                thresholds, cluster_counts
            )

            ax2.scatter(
                thresholds, cluster_counts, alpha=0.7, s=30, label="Data points"
            )
            if len(x_smooth) > 0:
                ax2.plot(
                    x_smooth,
                    y_smooth,
                    "b-",
                    linewidth=2,
                    alpha=0.8,
                    label="Exponential decay fit",
                )

            if len(inflection_x) > 0 and len(x_smooth) > 0:
                inflection_y = np.interp(inflection_x, x_smooth, y_smooth)
                ax2.scatter(
                    inflection_x,
                    inflection_y,
                    color="orange",
                    s=100,
                    marker="*",
                    zorder=6,
                    label=f"Key points ({len(inflection_x)})",
                )

            ax2.axvline(
                x=optimal_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold = {optimal_threshold:.6f}",
            )
            ax2.scatter(
                [optimal_threshold],
                [threshold_clustering["n_clusters"]],
                color="red",
                s=100,
                zorder=5,
                label=f"Selected ({threshold_clustering['n_clusters']} clusters)",
            )

            ax2.set_xlabel("nRMSD Threshold")
            ax2.set_ylabel("Number of Clusters")
            ax2.set_title("Threshold vs Cluster Count with Exponential Decay Fit")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig("clustering_analysis.png", dpi=300, bbox_inches="tight")
            print("Clustering analysis plots saved to clustering_analysis.png")

            try:
                plt.show()
            except Exception:
                print("Note: Could not display plot interactively, but saved to file")
        except ImportError:
            print(
                "Warning: matplotlib not available, skipping visualization",
                file=sys.stderr,
            )

    # Summary
    print(f"\nFound {len(all_threshold_data)} different clustering configurations")
    print(
        f"Threshold range: {all_threshold_data[0]['nrmsd_threshold']:.6f} to {all_threshold_data[-1]['nrmsd_threshold']:.6f}"
    )
    print(
        f"Cluster count range: {len(all_threshold_data[-1]['clusters'])} to {len(all_threshold_data[0]['clusters'])}"
    )

    print(f"\nClustering at threshold {optimal_threshold:.6f}:")
    print(f"  Number of clusters: {threshold_clustering['n_clusters']}")
    print(f"  Cluster sizes: {threshold_clustering['cluster_sizes']}")
    for i, cluster in enumerate(threshold_clustering["clusters"]):
        print(
            f"  Cluster {i + 1}: {cluster['representative']} + {len(cluster['members'])} members"
        )

    if args.output_json:
        output_data = {
            "all_thresholds": all_threshold_data,
            "threshold_clustering": threshold_clustering,
            "parameters": {
                "threshold": optimal_threshold,
                "threshold_source": "user-specified"
                if args.threshold is not None
                else "auto-detected",
                "rmsd_method": args.rmsd_method,
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nComprehensive clustering results saved to {args.output_json}")


# Approximate mode helper functions and workflow
# ----------------------------------------------------------------------
def _select_base_atoms(residue) -> List[Optional[np.ndarray]]:
    """
    Select four canonical base atoms for a nucleotide residue.

    Purines (A/G/DA/DG): N9, N3, N1, C5
    Pyrimidines (C/U/DC/DT): N1, O2, N3, C5

    If residue name is unknown, we try purine mapping first and, if incomplete,
    fall back to pyrimidine mapping. Returned list always has length 4 and may
    contain ``None`` when coordinates are missing.
    """
    purines = {"A", "G", "DA", "DG"}
    pyrimidines = {"C", "U", "DC", "DT"}

    def _coords_for(names: List[str]) -> List[Optional[np.ndarray]]:
        """Helper to fetch coordinates for a list of atom names."""
        return [
            (atom.coordinates if (atom := residue.find_atom(n)) is not None else None)
            for n in names
        ]

    if residue.residue_name in purines:
        return _coords_for(["N9", "N3", "N1", "C5"])

    if residue.residue_name in pyrimidines:
        return _coords_for(["N1", "O2", "N3", "C5"])

    # Unknown residue – attempt purine rule first, then pyrimidine
    coords = _coords_for(["N9", "N3", "N1", "C5"])
    if all(c is not None for c in coords):
        return coords
    return _coords_for(["N1", "O2", "N3", "C5"])


def featurize_structure(structure: Structure) -> np.ndarray:
    """
    Convert a Structure into a fixed-length feature vector.
    For n residues the length is 34 * n * (n-1) / 2.
    """
    residues = [r for r in structure.residues if r.is_nucleotide]
    n = len(residues)
    if n < 2:
        return np.zeros(0, dtype=np.float32)

    base_coords = [_select_base_atoms(r) for r in residues]
    feats: List[float] = []

    for i in range(n):
        ai = base_coords[i]
        for j in range(i + 1, n):
            aj = base_coords[j]

            # 16 distances
            for ci in ai:
                for cj in aj:
                    if ci is None or cj is None:
                        dist = 0.0
                    else:
                        dist = float(np.linalg.norm(ci - cj))
                    feats.append(dist)

            # 18 torsion features (sin, cos over 9 angles)
            a1 = ai[0]
            a4 = aj[0]
            for idx2 in range(1, 4):
                for idx3 in range(1, 4):
                    a2, a3 = ai[idx2], aj[idx3]
                    if any(x is None for x in (a1, a2, a3, a4)):
                        feats.extend([0.0, 1.0])
                    else:
                        angle = calculate_torsion_angle(a1, a2, a3, a4)
                        feats.extend([float(np.sin(angle)), float(np.cos(angle))])

    return np.asarray(feats, dtype=np.float32)


def run_approximate(structures: List[Structure], file_paths: List[Path], args) -> None:
    """
    Approximate mode: features → PCA → FAISS radius clustering.
    """
    print("\nRunning approximate mode (feature-based PCA + FAISS)")

    feature_vectors = [featurize_structure(s) for s in structures]
    feature_lengths = {len(v) for v in feature_vectors}
    if len(feature_lengths) != 1:
        print("Error: Inconsistent feature lengths among structures", file=sys.stderr)
        sys.exit(1)

    X = np.stack(feature_vectors).astype(np.float32)
    print(f"Feature matrix shape: {X.shape}")

    pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
    X_red = pca.fit_transform(X).astype(np.float32)
    d = X_red.shape[1]
    print(f"PCA reduced to {d} dimensions (95 % variance)")

    index = faiss.IndexFlatL2(d)
    index.add(X_red)
    radius_sq = args.radius**2

    visited: set[int] = set()
    clusters: List[List[int]] = []

    for idx in range(len(structures)):
        if idx in visited:
            continue
        D, I = index.search(X_red[idx : idx + 1], len(structures))
        cluster = [int(i) for dist, i in zip(D[0], I[0]) if dist <= radius_sq]
        clusters.append(cluster)
        visited.update(cluster)

    print(f"\nIdentified {len(clusters)} representatives with radius {args.radius}")
    for cluster in clusters:
        rep = cluster[0]
        redundants = cluster[1:]
        print(f"Representative: {file_paths[rep]}")
        for r in redundants:
            print(f"  Redundant: {file_paths[r]}")

    if args.output_json:
        out = {
            "parameters": {"mode": "approximate", "radius": args.radius},
            "clusters": [
                {
                    "representative": str(file_paths[c[0]]),
                    "members": [str(file_paths[m]) for m in c[1:]],
                }
                for c in clusters
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nApproximate clustering saved to {args.output_json}")

    return


# ----------------------------------------------------------------------


def find_all_thresholds_and_clusters(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray, file_paths: List[Path]
) -> List[dict]:
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
    List[dict]
        List of threshold cluster data
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

    return threshold_data


def find_structure_clusters(
    structures: List[Structure],
    file_paths: List[Path],
    cache: NRMSDCache,
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

    # Prepare all pairs, checking cache first
    cached_pairs = []
    compute_pairs = []

    for i, j in itertools.combinations(range(n_structures), 2):
        cached_value = cache.get(file_paths[i], file_paths[j], rmsd_method)
        if cached_value is not None:
            cached_pairs.append((i, j, cached_value))
        else:
            compute_pairs.append((i, j, nucleotide_lists[i], nucleotide_lists[j]))

    print(
        f"Found {len(cached_pairs)} cached values, computing {len(compute_pairs)} new values"
    )

    # Fill distance matrix with cached values
    for i, j, nrmsd_value in cached_pairs:
        distance_matrix[i, j] = nrmsd_value
        distance_matrix[j, i] = nrmsd_value

    # Process remaining pairs with progress bar and timing
    if compute_pairs:
        start_time = time.time()
        with ProcessPoolExecutor() as executor:
            futures_dict = {
                executor.submit(nrmsd_func, nucleotides_i, nucleotides_j): (i, j)
                for i, j, nucleotides_i, nucleotides_j in compute_pairs
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

                # Cache the computed value
                cache.set(file_paths[i], file_paths[j], rmsd_method, nrmsd_value)

        end_time = time.time()
        computation_time = end_time - start_time

        print(f"RMSD computation completed in {computation_time:.2f} seconds")
        if rmsd_method == "validate":
            print(
                "Note: Validation mode tests all methods, so timing includes overhead from multiple calculations"
            )

        # Fill the distance matrix with computed values
        for i, j, nrmsd in results:
            distance_matrix[i, j] = nrmsd
            distance_matrix[j, i] = nrmsd

    # Save cache after all computations
    cache.save_cache()

    # Convert to condensed distance matrix for scipy
    condensed_distances = squareform(distance_matrix)

    # Perform hierarchical clustering with complete linkage
    linkage_matrix = linkage(condensed_distances, method="complete")

    # Return distance matrix for further processing
    return distance_matrix


def get_clustering_at_threshold(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    threshold: float,
) -> dict:
    """
    Get clustering results at a specific threshold.

    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    distance_matrix : np.ndarray
        Square distance matrix
    file_paths : List[Path]
        List of file paths corresponding to structures
    threshold : float
        nRMSD threshold for clustering

    Returns:
    --------
    dict
        Dictionary with clustering information at the given threshold
    """
    # Get cluster assignments at this threshold
    labels = fcluster(linkage_matrix, threshold, criterion="distance")
    n_clusters = len(np.unique(labels))

    # Group structure indices by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    cluster_sizes = [len(cluster) for cluster in clusters.values()]
    cluster_sizes.sort(reverse=True)

    # Find medoids for each cluster
    medoids = find_cluster_medoids(list(clusters.values()), distance_matrix)

    # Create result data
    result = {
        "nrmsd_threshold": float(threshold),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "clusters": [],
    }

    for cluster_indices, medoid_idx in zip(clusters.values(), medoids):
        representative = str(file_paths[medoid_idx])
        members = [str(file_paths[idx]) for idx in cluster_indices if idx != medoid_idx]

        result["clusters"].append(
            {"representative": representative, "members": members}
        )

    return result


def exponential_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Exponential decay function: y = a * exp(-b * x) + c

    Parameters:
    -----------
    x : np.ndarray
        Input values
    a : float
        Amplitude parameter
    b : float
        Decay rate parameter
    c : float
        Offset parameter

    Returns:
    --------
    np.ndarray
        Function values
    """
    return a * np.exp(-b * x) + c


def fit_exponential_decay(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit exponential decay function to data and find inflection points.

    Parameters:
    -----------
    x : np.ndarray
        X coordinates (thresholds)
    y : np.ndarray
        Y coordinates (cluster counts)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (x_smooth, y_smooth, inflection_x) where:
        - x_smooth: smooth x values for plotting the fitted curve
        - y_smooth: smooth y values for plotting the fitted curve
        - inflection_x: x coordinates of inflection points
    """
    if len(x) < 4:
        return x, y, np.array([])

    # Sort data by x values
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    try:
        # Initial parameter guess
        a_guess = y_sorted.max() - y_sorted.min()
        b_guess = 1.0
        c_guess = y_sorted.min()

        # Fit exponential decay
        popt, _ = curve_fit(
            exponential_decay,
            x_sorted,
            y_sorted,
            p0=[a_guess, b_guess, c_guess],
            maxfev=5000,
        )

        a_fit, b_fit, c_fit = popt

        # Generate smooth curve for plotting
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_smooth = exponential_decay(x_smooth, a_fit, b_fit, c_fit)

        # For exponential decay y = a*exp(-b*x) + c, the second derivative is:
        # y'' = a*b^2*exp(-b*x)
        # Since a > 0 and b > 0 for decay, y'' > 0 always, so no inflection points
        # However, we can find the point of maximum curvature (steepest decline)
        # This occurs where the first derivative is most negative
        # y' = -a*b*exp(-b*x), which is most negative at x = 0
        # But we'll look for the point where the rate of change is fastest within our data range

        # For exponential decay, identify the "knee" point where the curve
        # transitions from steep to gradual decline
        # This is often around x = 1/b in the exponential decay
        knee_x = 1.0 / b_fit if b_fit > 0 else None

        # Also find the point where the second derivative is maximum
        # (point of maximum curvature, excluding edges)
        second_deriv_vals = a_fit * (b_fit**2) * np.exp(-b_fit * x_smooth)

        # Exclude edge points (first and last 10% of data)
        edge_margin = int(0.1 * len(x_smooth))
        if edge_margin < 1:
            edge_margin = 1

        # Find maximum curvature point excluding edges
        max_curvature_idx = (
            np.argmax(second_deriv_vals[edge_margin:-edge_margin]) + edge_margin
        )
        max_curvature_x = x_smooth[max_curvature_idx]

        inflection_points = []

        # Add knee point if it's within data range and not at edges
        if knee_x is not None and x_sorted.min() + 0.1 * (
            x_sorted.max() - x_sorted.min()
        ) <= knee_x <= x_sorted.max() - 0.1 * (x_sorted.max() - x_sorted.min()):
            inflection_points.append(knee_x)

        # Add maximum curvature point if it's meaningful and different from knee
        if x_sorted.min() + 0.1 * (
            x_sorted.max() - x_sorted.min()
        ) <= max_curvature_x <= x_sorted.max() - 0.1 * (
            x_sorted.max() - x_sorted.min()
        ) and (
            not inflection_points
            or abs(max_curvature_x - inflection_points[0])
            > 0.05 * (x_sorted.max() - x_sorted.min())
        ):
            inflection_points.append(max_curvature_x)

        inflection_x = np.array(inflection_points)

        print(
            f"Exponential decay fit: y = {a_fit:.3f} * exp(-{b_fit:.3f} * x) + {c_fit:.3f}"
        )

        return x_smooth, y_smooth, inflection_x

    except Exception as e:
        print(f"Warning: Exponential decay fitting failed: {e}")
        return x, y, np.array([])


def determine_optimal_threshold(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray
) -> float:
    """
    Determine optimal threshold from exponential decay inflection point.

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
    # Extract merge distances from linkage matrix
    merge_distances = linkage_matrix[:, 2]
    valid_thresholds = np.sort(merge_distances)

    # Calculate cluster counts for each threshold
    cluster_counts = []
    for threshold in valid_thresholds:
        labels = fcluster(linkage_matrix, threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        cluster_counts.append(n_clusters)

    thresholds = np.array(valid_thresholds)
    cluster_counts = np.array(cluster_counts)

    # Fit exponential decay and find inflection points
    x_smooth, y_smooth, inflection_x = fit_exponential_decay(thresholds, cluster_counts)

    if len(inflection_x) > 0:
        # Use the first inflection point as the optimal threshold
        optimal_threshold = inflection_x[0]
        print(f"Auto-detected optimal threshold: {optimal_threshold:.6f}")
        return optimal_threshold
    else:
        # Fallback to a reasonable default if no inflection points found
        fallback_threshold = 0.1
        print(
            f"No inflection points found, using fallback threshold: {fallback_threshold}"
        )
        return fallback_threshold


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

    # Switch workflow based on requested mode
    if args.mode == "approximate":
        run_approximate(structures, valid_files, args)
        return
    else:
        run_exact(structures, valid_files, args)
        return


if __name__ == "__main__":
    main()
