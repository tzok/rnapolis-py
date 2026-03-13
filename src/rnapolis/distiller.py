import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the RNA structure clustering CLI.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
Cluster RNA 3D structures by geometric similarity.

Two modes are available:

  approximate (default)  Fast feature-based clustering. Structures are
                         featurised with inter-base distances and torsion
                         angles, projected via PCA, and clustered in the
                         reduced space.

  exact                  Rigorous all-vs-all nRMSD comparison. Slower but
                         produces publication-quality distance matrices.

Three clustering methods can be combined with either mode:

  hierarchical (default) Complete-linkage agglomerative clustering with
                         auto-detected or user-specified threshold.
  affinity-propagation   Message-passing algorithm that discovers exemplars
                         and the number of clusters automatically.
  facility-location      Submodular optimisation that selects exactly N
                         maximally representative structures (requires
                         --n-representatives).

examples:
  # Auto-detect clusters (approximate + hierarchical, the default)
  distiller *.cif

  # Pipe file paths from find
  find structures/ -name '*.pdb' | distiller

  # Exact mode with auto-detected threshold
  distiller --mode exact *.cif

  # Exact mode with a specific nRMSD threshold
  distiller --mode exact --threshold 0.15 *.cif

  # Affinity propagation (let AP choose cluster count)
  distiller --method affinity-propagation *.cif

  # Select exactly 5 representatives via facility location
  distiller --n-representatives 5 *.cif

  # Approximate mode with explicit radii
  distiller --radius 1.0 --radius 4.0 *.cif

  # Save results and produce plots
  distiller --output-json results.json --visualize *.cif""",
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="input mmCIF or PDB files (use '-' or omit to read paths from stdin)",
    )

    # -- Output options ---------------------------------------------------
    output_group = parser.add_argument_group("output options")

    output_group.add_argument(
        "--output-json",
        type=str,
        metavar="FILE",
        help="save clustering results to a JSON file",
    )

    output_group.add_argument(
        "--visualize",
        action="store_true",
        help="show dendrogram / MDS scatter plots of the clustering",
    )

    # -- Mode & method ----------------------------------------------------
    mode_group = parser.add_argument_group("mode & method")

    mode_group.add_argument(
        "--mode",
        choices=["exact", "approximate"],
        default="approximate",
        help=(
            "distance computation strategy (default: approximate). "
            "'approximate' uses PCA + FAISS; "
            "'exact' uses rigorous pairwise nRMSD"
        ),
    )

    mode_group.add_argument(
        "--method",
        choices=["hierarchical", "affinity-propagation"],
        default="hierarchical",
        help=(
            "clustering algorithm (default: hierarchical). "
            "ignored when --n-representatives is set"
        ),
    )

    mode_group.add_argument(
        "--n-representatives",
        type=int,
        default=None,
        metavar="N",
        help=(
            "select exactly N representatives via submodular facility "
            "location (overrides --method and --threshold)"
        ),
    )

    # -- Hierarchical options ---------------------------------------------
    hier_group = parser.add_argument_group("hierarchical clustering options")

    hier_group.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "nRMSD distance threshold (exact mode). "
            "if omitted, auto-detected from the exponential-decay knee"
        ),
    )

    hier_group.add_argument(
        "--radius",
        type=float,
        action="append",
        default=None,
        help=(
            "PCA-space radius for approximate mode. "
            "can be given multiple times; "
            "if omitted, auto-detected like --threshold"
        ),
    )

    # -- Affinity propagation options -------------------------------------
    ap_group = parser.add_argument_group("affinity propagation options")

    ap_group.add_argument(
        "--preference",
        type=float,
        default=None,
        help=(
            "AP preference — controls cluster count "
            "(default: median of similarities; "
            "more negative = fewer clusters)"
        ),
    )

    ap_group.add_argument(
        "--damping",
        type=float,
        default=0.9,
        help=(
            "AP damping factor in [0.5, 1.0) "
            "(default: 0.9; higher = more stable convergence)"
        ),
    )

    # -- Exact-mode options -----------------------------------------------
    exact_group = parser.add_argument_group("exact mode options")

    exact_group.add_argument(
        "--rmsd-method",
        type=str,
        choices=["quaternions", "svd", "qcp", "validate"],
        default="quaternions",
        help=(
            "nRMSD algorithm (default: quaternions). "
            "'validate' runs all three and checks agreement"
        ),
    )

    exact_group.add_argument(
        "--cache-file",
        type=str,
        default="nrmsd_cache.json",
        metavar="FILE",
        help="file for caching computed nRMSD values (default: nrmsd_cache.json)",
    )

    exact_group.add_argument(
        "--cache-save-interval",
        type=int,
        default=100,
        metavar="N",
        help="save cache every N computations (default: 100)",
    )

    return parser.parse_args()


class NRMSDCache:
    """Cache for nRMSD values keyed by file metadata and RMSD method."""

    def __init__(self, cache_file: str, save_interval: int = 100):
        """Initialize the nRMSD cache and load existing values from disk.

        Args:
            cache_file (str): Path to the JSON file used for persisting cache.
            save_interval (int): Number of new computations between automatic cache saves.
        """
        self.cache_file = cache_file
        self.save_interval = save_interval
        self.cache: Dict[str, float] = {}
        self.computation_count = 0
        self.load_cache()

    def _get_file_key(self, file_path: Path) -> str:
        """Build a unique key for a file using its absolute path and stat information.

        Args:
            file_path (Path): Path to the structure file.

        Returns:
            str: Stable identifier for this file including mtime and size.
        """
        stat = file_path.stat()
        return f"{file_path.absolute()}:{stat.st_mtime}:{stat.st_size}"

    def _get_pair_key(self, file1: Path, file2: Path, rmsd_method: str) -> str:
        """Build a unique cache key for a pair of files and an RMSD method.

        Args:
            file1 (Path): Path to the first structure file.
            file2 (Path): Path to the second structure file.
            rmsd_method (str): RMSD method name used for this comparison.

        Returns:
            str: Hash key identifying this file pair and method.
        """
        key1 = self._get_file_key(file1)
        key2 = self._get_file_key(file2)
        # Ensure consistent ordering
        if key1 > key2:
            key1, key2 = key2, key1
        combined = f"{key1}|{key2}|{rmsd_method}"
        # Use hash to keep keys manageable
        return hashlib.md5(combined.encode()).hexdigest()

    def load_cache(self):
        """Load cached nRMSD values from disk if a cache file exists."""
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
        """Persist the cache to disk as JSON.

        Args:
            silent (bool): If True, suppress confirmation message.
        """
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
        """Retrieve a cached nRMSD value for the given file pair and method.

        Args:
            file1 (Path): Path to the first structure file.
            file2 (Path): Path to the second structure file.
            rmsd_method (str): RMSD method name used for this comparison.

        Returns:
            Cached nRMSD value if present, otherwise None.
        """
        key = self._get_pair_key(file1, file2, rmsd_method)
        return self.cache.get(key)

    def set(self, file1: Path, file2: Path, rmsd_method: str, value: float):
        """Store a newly computed nRMSD value in the cache.

        Args:
            file1 (Path): Path to the first structure file.
            file2 (Path): Path to the second structure file.
            rmsd_method (str): RMSD method name used for this comparison.
            value (float): Computed nRMSD value.
        """
        key = self._get_pair_key(file1, file2, rmsd_method)
        self.cache[key] = value
        self.computation_count += 1

        # Save periodically (silently to avoid disrupting progress bar)
        if self.computation_count % self.save_interval == 0:
            self.save_cache(silent=True)


def validate_input_files(files: List[Path]) -> List[Path]:
    """Filter and validate input files by existence and supported extension.

    Args:
        files (List[Path]): List of input file paths provided by the user.

    Returns:
        List of existing files with recognized structure extensions.
    """
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
    """Parse a PDB or mmCIF file into a Structure object.

    Args:
        file_path (Path): Path to the structure file.

    Returns:
        Structure: Parsed structure built from atom coordinates.

    Raises:
        Exception: Propagates parsing errors after logging.
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
    """Check that all parsed structures have the same number of nucleotides.

    Args:
        structures (List[Structure]): Parsed structures to validate.
        file_paths (List[Path]): Paths corresponding to the structures.

    Raises:
        SystemExit: If any structure has a different nucleotide count.
    """
    nucleotide_counts = []

    for structure, file_path in tqdm(
        zip(structures, file_paths),
        total=len(structures),
        desc="Validating nucleotide counts",
        unit="structure",
    ):
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


# ======================================================================
# Shared clustering helpers (used by both exact and approximate modes)
# ======================================================================


def cluster_affinity_propagation(
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    preference: Optional[float] = None,
    damping: float = 0.9,
) -> dict:
    """Cluster structures using Affinity Propagation on a distance matrix.

    The distance matrix is converted to a similarity matrix via
    ``similarity = -(distance ** 2)`` and passed to scikit-learn's
    :class:`~sklearn.cluster.AffinityPropagation`.  Exemplars discovered
    by the algorithm serve directly as cluster representatives.

    Args:
        distance_matrix: Square symmetric distance matrix (L2 or nRMSD).
        file_paths: Paths corresponding to the structures.
        preference: AP preference (controls cluster count).
            ``None`` uses the median of the similarity values.
        damping: Damping factor in ``[0.5, 1.0)`` (default 0.9).

    Returns:
        Clustering result dict with the same schema as
        :func:`get_clustering_at_threshold`.
    """
    similarity = -(distance_matrix**2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=damping,
            preference=preference,
            max_iter=300,
            convergence_iter=15,
            random_state=0,
        )
        ap.fit(similarity)

    for w in caught:
        if "did not converge" in str(w.message):
            print(
                f"Warning: Affinity Propagation did not converge after "
                f"{ap.n_iter_} iterations. Consider increasing --damping "
                f"(current: {damping}).",
                file=sys.stderr,
            )

    exemplar_indices = ap.cluster_centers_indices_
    labels = ap.labels_

    if exemplar_indices is None or labels is None:
        print(
            "Error: Affinity Propagation produced degenerate results. "
            "Try adjusting --damping or --preference.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_clusters = len(exemplar_indices)

    print(f"Affinity Propagation found {n_clusters} clusters (exemplars)")

    # Build cluster membership lists
    clusters: dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    cluster_sizes = sorted(
        (len(members) for members in clusters.values()), reverse=True
    )

    result: dict = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "clusters": [],
    }

    for cluster_id in range(n_clusters):
        exemplar_idx = int(exemplar_indices[cluster_id])
        member_indices = clusters.get(cluster_id, [])
        members = [
            str(file_paths[idx]) for idx in member_indices if idx != exemplar_idx
        ]
        result["clusters"].append(
            {"representative": str(file_paths[exemplar_idx]), "members": members}
        )

    return result


def cluster_facility_location(
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    n_representatives: int,
) -> dict:
    """Select *n_representatives* using submodular facility location.

    Uses the *apricot-select* library to greedily pick a maximally
    representative subset.  The distance matrix is converted to a
    non-negative similarity matrix via ``max(d) - d``.

    Args:
        distance_matrix: Square symmetric distance matrix.
        file_paths: Paths corresponding to the structures.
        n_representatives: Exact number of representatives to select.

    Returns:
        Clustering result dict.  Each "cluster" groups a representative
        with the non-selected structures nearest to it.
    """
    try:
        from apricot import FacilityLocationSelection
    except ImportError:
        print(
            "Error: apricot-select is required for --n-representatives. "
            "Install it with:  pip install apricot-select",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(file_paths)
    if n_representatives >= n:
        print(
            f"Warning: --n-representatives ({n_representatives}) >= number of "
            f"structures ({n}); selecting all structures.",
            file=sys.stderr,
        )
        n_representatives = n

    # apricot requires a non-negative similarity matrix
    similarity = distance_matrix.max() - distance_matrix
    np.fill_diagonal(similarity, 0.0)

    selector = FacilityLocationSelection(
        n_samples=n_representatives,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(similarity)

    ranking = selector.ranking  # ordered indices of selected representatives
    gains = selector.gains  # marginal gain at each selection step

    print(
        f"Facility location selected {n_representatives} representatives "
        f"(total gain: {gains.sum():.4f})"
    )

    # Assign every non-selected structure to its nearest representative
    selected_set = set(int(r) for r in ranking)
    rep_indices = np.array(ranking, dtype=int)

    # For each structure, find the nearest selected representative
    nearest_rep: dict[int, List[int]] = {int(r): [] for r in ranking}
    for idx in range(n):
        if idx in selected_set:
            continue
        dists_to_reps = distance_matrix[idx, rep_indices]
        best = rep_indices[int(np.argmin(dists_to_reps))]
        nearest_rep[int(best)].append(idx)

    cluster_sizes = sorted(
        (1 + len(members) for members in nearest_rep.values()), reverse=True
    )

    result: dict = {
        "n_clusters": n_representatives,
        "cluster_sizes": cluster_sizes,
        "clusters": [],
        "ranking": [int(r) for r in ranking],
        "gains": [float(g) for g in gains],
    }

    for r_idx in ranking:
        r_idx = int(r_idx)
        members = [str(file_paths[m]) for m in nearest_rep[r_idx]]
        result["clusters"].append(
            {"representative": str(file_paths[r_idx]), "members": members}
        )

    return result


def _visualize_flat_clustering(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    exemplar_indices: np.ndarray,
    title: str,
    output_file: str,
    gains: Optional[List[float]] = None,
) -> None:
    """Produce a visualization for a flat (non-hierarchical) clustering.

    Panel 1 is a 2-D MDS projection coloured by cluster label with
    exemplars/representatives highlighted.  When *gains* are supplied
    (facility location mode), panel 2 shows the marginal-gain curve.

    Args:
        distance_matrix: Square distance matrix used for MDS.
        labels: Cluster label for each structure (length N).
        exemplar_indices: Indices of representative structures.
        title: Figure super-title.
        output_file: Path to save the PNG.
        gains: Optional marginal gains (facility location only).
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import MDS
    except ImportError:
        print(
            "Warning: matplotlib/sklearn not available, skipping visualization",
            file=sys.stderr,
        )
        return

    n_panels = 2 if gains else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: MDS scatter coloured by cluster ---
    ax = axes[0]
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(distance_matrix)

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab20", alpha=0.6, s=40
    )
    rep_coords = coords[exemplar_indices]
    ax.scatter(
        rep_coords[:, 0],
        rep_coords[:, 1],
        c="red",
        s=120,
        marker="*",
        zorder=5,
        edgecolors="black",
        linewidths=0.5,
        label=f"Representatives ({len(exemplar_indices)})",
    )
    ax.set_title("MDS Projection of Structures")
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Marginal gains (facility location only) ---
    if gains and len(axes) > 1:
        ax2 = axes[1]
        ax2.bar(range(1, len(gains) + 1), gains, color="steelblue", alpha=0.8)
        ax2.set_xlabel("Selection Order")
        ax2.set_ylabel("Marginal Gain")
        ax2.set_title("Facility Location Marginal Gains")
        ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")

    try:
        plt.show()
    except Exception:
        print("Note: Could not display plot interactively, but saved to file")


def _print_clustering_summary(clustering: dict) -> None:
    """Print a human-readable summary of a flat clustering result.

    Args:
        clustering: Dict with keys ``n_clusters``, ``cluster_sizes``,
            and ``clusters`` (list of representative/members dicts).
    """
    print(f"\n  Number of clusters: {clustering['n_clusters']}")
    print(f"  Cluster sizes: {clustering['cluster_sizes']}")
    for i, cluster in enumerate(clustering["clusters"]):
        print(
            f"  Cluster {i + 1}: {cluster['representative']} "
            f"+ {len(cluster['members'])} members"
        )


def _labels_from_clustering(clustering: dict, file_paths: List[Path]) -> np.ndarray:
    """Reconstruct a per-structure label array from a clustering dict.

    Args:
        clustering: Dict whose ``clusters`` list contains
            ``representative`` and ``members`` string paths.
        file_paths: Ordered list of paths (defines index mapping).

    Returns:
        Integer label array of length ``len(file_paths)``.
    """
    path_to_idx = {str(p): i for i, p in enumerate(file_paths)}
    labels = np.full(len(file_paths), -1, dtype=int)
    for cluster_id, cluster in enumerate(clustering["clusters"]):
        rep_idx = path_to_idx.get(cluster["representative"])
        if rep_idx is not None:
            labels[rep_idx] = cluster_id
        for member_path in cluster["members"]:
            member_idx = path_to_idx.get(member_path)
            if member_idx is not None:
                labels[member_idx] = cluster_id
    return labels


def _exemplar_indices_from_clustering(
    clustering: dict, file_paths: List[Path]
) -> np.ndarray:
    """Extract exemplar (representative) indices from a clustering dict.

    Args:
        clustering: Dict whose ``clusters`` list contains
            ``representative`` string paths.
        file_paths: Ordered list of paths (defines index mapping).

    Returns:
        Integer array of exemplar indices.
    """
    path_to_idx = {str(p): i for i, p in enumerate(file_paths)}
    indices = []
    for cluster in clustering["clusters"]:
        idx = path_to_idx.get(cluster["representative"])
        if idx is not None:
            indices.append(idx)
    return np.array(indices, dtype=int)


def _write_json(
    output_path: str,
    clustering: dict,
    *,
    mode: str,
    method: str,
    n_structures: int,
    **extra_params,
) -> None:
    """Write clustering results to a JSON file with metadata.

    Args:
        output_path: Destination file path.
        clustering: Clustering result dict (n_clusters, clusters, …).
        mode: ``"exact"`` or ``"approximate"``.
        method: Clustering method name (e.g. ``"affinity-propagation"``).
        n_structures: Total number of input structures.
        **extra_params: Additional parameters to include in the
            ``parameters`` section (e.g. ``preference``, ``damping``).
    """
    output_data = {
        "clustering": clustering,
        "parameters": {
            "mode": mode,
            "method": method,
            "n_structures": n_structures,
            **extra_params,
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nClustering results saved to {output_path}")


# ----------------------------------------------------------------------


def run_exact(
    structures: List[Structure], valid_files: List[Path], args: argparse.Namespace
) -> None:
    """Run exact nRMSD-based clustering workflow and optional visualization.

    Args:
        structures (List[Structure]): List of parsed structures.
        valid_files (List[Path]): File paths corresponding to the structures.
        args: Parsed CLI arguments controlling mode, cache and visualization.
    """
    # Initialize cache
    print("\nInitializing nRMSD cache...")
    cache = NRMSDCache(args.cache_file, args.cache_save_interval)

    # Compute distance matrix
    print("\nComputing distance matrix...")
    distance_matrix = find_structure_clusters(
        structures, valid_files, cache, args.visualize, args.rmsd_method
    )

    # ------------------------------------------------------------------
    # Facility location (--n-representatives) takes priority
    # ------------------------------------------------------------------
    if args.n_representatives is not None:
        clustering = cluster_facility_location(
            distance_matrix, valid_files, args.n_representatives
        )
        _print_clustering_summary(clustering)

        if args.visualize:
            labels = _labels_from_clustering(clustering, valid_files)
            exemplar_indices = np.array(clustering["ranking"])
            _visualize_flat_clustering(
                distance_matrix,
                labels,
                exemplar_indices,
                "Facility Location Selection (Exact nRMSD)",
                "facility_location_analysis.png",
                gains=clustering.get("gains"),
            )

        if args.output_json:
            _write_json(
                args.output_json,
                clustering,
                mode="exact",
                method="facility-location",
                n_structures=len(structures),
                rmsd_method=args.rmsd_method,
                n_representatives=args.n_representatives,
            )
        return

    # ------------------------------------------------------------------
    # Affinity propagation (--method affinity-propagation)
    # ------------------------------------------------------------------
    if args.method == "affinity-propagation":
        clustering = cluster_affinity_propagation(
            distance_matrix, valid_files, args.preference, args.damping
        )
        _print_clustering_summary(clustering)

        if args.visualize:
            labels = _labels_from_clustering(clustering, valid_files)
            exemplar_indices = _exemplar_indices_from_clustering(
                clustering, valid_files
            )
            _visualize_flat_clustering(
                distance_matrix,
                labels,
                exemplar_indices,
                "Affinity Propagation Clustering (Exact nRMSD)",
                "clustering_analysis.png",
            )

        if args.output_json:
            _write_json(
                args.output_json,
                clustering,
                mode="exact",
                method="affinity-propagation",
                n_structures=len(structures),
                rmsd_method=args.rmsd_method,
                preference=args.preference,
                damping=args.damping,
            )
        return

    # ------------------------------------------------------------------
    # Hierarchical clustering (default)
    # ------------------------------------------------------------------
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
                "method": "hierarchical",
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nComprehensive clustering results saved to {args.output_json}")


# Approximate mode helper functions and workflow
# ----------------------------------------------------------------------
def _select_base_atoms(residue) -> List[Optional[np.ndarray]]:
    """Select four canonical base atoms for a nucleotide residue.

    For purines (A/G/DA/DG) uses N9, N3, N1, C5.
    For pyrimidines (C/U/DC/DT) uses N1, O2, N3, C5.
    Falls back between schemes for unknown residue names.

    Args:
        residue: Residue3D-like object providing `residue_name` and `find_atom()`.

    Returns:
        List[Optional[np.ndarray]]: List of up to four atom coordinates, with None for missing atoms.
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
    """Convert a structure into a fixed-length feature vector.

    For n nucleotide residues the feature length is 34 * n * (n - 1) / 2,
    combining inter-base distances and torsion-based sine/cosine terms.

    Args:
        structure (Structure): Structure whose nucleotide residues will be featurized.

    Returns:
        np.ndarray: 1D float32 array of pairwise geometric features.
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


def run_approximate_multiple(
    structures: List[Structure],
    file_paths: List[Path],
    radii: Optional[List[float]],
    output_json: Optional[str],
    visualize: bool = False,
    method: str = "hierarchical",
    preference: Optional[float] = None,
    damping: float = 0.9,
    n_representatives: Optional[int] = None,
) -> None:
    """Run approximate PCA + FAISS-based redundancy detection.

    When *radii* are provided explicitly, the original greedy FAISS scan is
    used for each radius value.  When *radii* is ``None`` (the default), the
    optimal radius is auto-detected by computing a pairwise L2 distance
    matrix in PCA-reduced space, building a hierarchical linkage tree, and
    selecting the "knee" of the threshold-vs-cluster-count curve (same
    strategy used by exact mode for nRMSD thresholds).

    Args:
        structures: Parsed structures to analyse.
        file_paths: Paths corresponding to the structures.
        radii: Optional radii in reduced space.  ``None`` triggers auto-detect.
        output_json: Optional path to save clustering summary as JSON.
        visualize: If ``True``, produce a dendrogram + threshold-vs-cluster
            plot (only used in auto-detect mode).
        method: Clustering method (``"hierarchical"`` or
            ``"affinity-propagation"``).
        preference: AP preference value (only for affinity-propagation).
        damping: AP damping factor (only for affinity-propagation).
        n_representatives: If set, use facility location selection instead
            of threshold-based clustering.
    """
    # ------------------------------------------------------------------
    # 0. Handle --radius + --method conflict
    # ------------------------------------------------------------------
    if radii is not None and method == "affinity-propagation":
        warnings.warn(
            "--radius is incompatible with --method affinity-propagation; "
            "ignoring --method and using greedy FAISS scan for specified radii.",
            stacklevel=2,
        )
        method = "hierarchical"

    # ------------------------------------------------------------------
    # 1. Feature extraction
    # ------------------------------------------------------------------
    print("\nRunning approximate mode (feature-based PCA + FAISS)")
    feature_vectors = [
        featurize_structure(s)
        for s in tqdm(structures, desc="Featurizing", unit="structure")
    ]
    feature_lengths = {len(v) for v in feature_vectors}
    if len(feature_lengths) != 1:
        print("Error: Inconsistent feature lengths among structures", file=sys.stderr)
        sys.exit(1)

    X = np.stack(feature_vectors).astype(np.float32)
    print(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------
    # 2. PCA transformation (fit once)
    # ------------------------------------------------------------------
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
    X_red = pca.fit_transform(X).astype(np.float32)
    d = X_red.shape[1]
    print(f"PCA reduced to {d} dimensions (95 % variance)")

    # ==================================================================
    # Branch A – auto-detect radius via hierarchical clustering
    #            (also used for AP and facility location in approx mode)
    # ==================================================================
    if radii is None:
        _run_approximate_auto(
            X_red,
            structures,
            file_paths,
            output_json,
            visualize,
            method=method,
            preference=preference,
            damping=damping,
            n_representatives=n_representatives,
        )
        return

    # ==================================================================
    # Branch B – user-supplied radii → greedy FAISS scan (original logic)
    # ==================================================================
    if not radii:
        print("Error: No radius values supplied", file=sys.stderr)
        sys.exit(1)

    index = faiss.IndexFlatL2(d)
    index.add(X_red)

    results_for_json: List[dict] = []
    for radius in radii:
        radius_sq = radius**2
        visited: set[int] = set()
        clusters: List[List[int]] = []

        for idx in range(len(structures)):
            if idx in visited:
                continue
            D, I = index.search(X_red[idx : idx + 1], len(structures))
            cluster = [int(i) for dist, i in zip(D[0], I[0]) if dist <= radius_sq]
            clusters.append(cluster)
            visited.update(cluster)

        print(f"\nIdentified {len(clusters)} representatives with radius {radius}")
        if output_json is None:
            for cluster in clusters:
                rep = cluster[0]
                redundants = cluster[1:]
                print(f"Representative: {file_paths[rep]}")
                for r in redundants:
                    print(f"  Redundant: {file_paths[r]}")

        if output_json is not None:
            results_for_json.append(
                {
                    "radius": radius,
                    "n_clusters": len(clusters),
                    "clusters": [
                        {
                            "representative": str(file_paths[c[0]]),
                            "members": [str(file_paths[m]) for m in c[1:]],
                        }
                        for c in clusters
                    ],
                }
            )

    # Write combined JSON once after processing all radii
    if output_json and results_for_json:
        combined = {
            "parameters": {
                "mode": "approximate",
                "radii": radii,
                "n_structures": len(structures),
            },
            "results": results_for_json,
        }
        with open(output_json, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nApproximate clustering for all radii saved to {output_json}")

    return


def _run_approximate_auto(
    X_red: np.ndarray,
    structures: List[Structure],
    file_paths: List[Path],
    output_json: Optional[str],
    visualize: bool,
    method: str = "hierarchical",
    preference: Optional[float] = None,
    damping: float = 0.9,
    n_representatives: Optional[int] = None,
) -> None:
    """Auto-detect optimal radius and cluster using hierarchical linkage.

    This mirrors the strategy used by exact mode: build a complete-linkage
    dendrogram over pairwise L2 distances in PCA-reduced space, enumerate
    every merge distance (each one splits the set into progressively fewer
    clusters), fit an exponential decay to the threshold-vs-cluster-count
    curve, and pick the "knee" as the optimal radius.

    When *n_representatives* or *method="affinity-propagation"* is set,
    the pairwise L2 distance matrix is still computed but the final
    clustering step uses the corresponding algorithm instead of
    hierarchical linkage.

    Args:
        X_red: PCA-reduced feature matrix (N × d, float32).
        structures: Parsed structures (used only for length).
        file_paths: Paths corresponding to the structures.
        output_json: Optional path to save clustering summary as JSON.
        visualize: If ``True``, produce clustering plots.
        method: Clustering method (``"hierarchical"`` or
            ``"affinity-propagation"``).
        preference: AP preference value.
        damping: AP damping factor.
        n_representatives: If set, use facility location selection.
    """
    n = len(structures)
    print(f"\nAuto-detecting optimal radius from {n} structures …")

    # ------------------------------------------------------------------
    # 1. Pairwise L2 distance matrix
    # ------------------------------------------------------------------
    condensed = pdist(X_red.astype(np.float64), metric="euclidean")
    distance_matrix = squareform(condensed)
    print(f"Pairwise L2 distance matrix computed ({n}×{n})")

    # ------------------------------------------------------------------
    # Facility location (--n-representatives) takes priority
    # ------------------------------------------------------------------
    if n_representatives is not None:
        clustering = cluster_facility_location(
            distance_matrix, file_paths, n_representatives
        )
        _print_clustering_summary(clustering)

        if visualize:
            labels = _labels_from_clustering(clustering, file_paths)
            exemplar_indices = np.array(clustering["ranking"])
            _visualize_flat_clustering(
                distance_matrix,
                labels,
                exemplar_indices,
                "Facility Location Selection (Approximate PCA L2)",
                "facility_location_analysis.png",
                gains=clustering.get("gains"),
            )

        if output_json:
            _write_json(
                output_json,
                clustering,
                mode="approximate",
                method="facility-location",
                n_structures=n,
                n_representatives=n_representatives,
            )
        return

    # ------------------------------------------------------------------
    # Affinity propagation (--method affinity-propagation)
    # ------------------------------------------------------------------
    if method == "affinity-propagation":
        clustering = cluster_affinity_propagation(
            distance_matrix, file_paths, preference, damping
        )
        _print_clustering_summary(clustering)

        if visualize:
            labels = _labels_from_clustering(clustering, file_paths)
            exemplar_indices = _exemplar_indices_from_clustering(clustering, file_paths)
            _visualize_flat_clustering(
                distance_matrix,
                labels,
                exemplar_indices,
                "Affinity Propagation Clustering (Approximate PCA L2)",
                "clustering_analysis.png",
            )

        if output_json:
            _write_json(
                output_json,
                clustering,
                mode="approximate",
                method="affinity-propagation",
                n_structures=n,
                preference=preference,
                damping=damping,
            )
        return

    # ------------------------------------------------------------------
    # Hierarchical clustering (default)
    # ------------------------------------------------------------------
    linkage_matrix = linkage(condensed, method="complete")

    # ------------------------------------------------------------------
    # 3. Auto-detect optimal radius (reuse exact mode's logic)
    # ------------------------------------------------------------------
    optimal_radius = determine_optimal_threshold(distance_matrix, linkage_matrix)

    # ------------------------------------------------------------------
    # 4. Enumerate all merge-distance clusterings
    # ------------------------------------------------------------------
    all_threshold_data = find_all_thresholds_and_clusters(
        distance_matrix, linkage_matrix, file_paths
    )
    threshold_clustering = get_clustering_at_threshold(
        linkage_matrix, distance_matrix, file_paths, optimal_radius
    )

    # ------------------------------------------------------------------
    # 5. Visualisation (mirrors exact mode's two-panel plot)
    # ------------------------------------------------------------------
    if visualize:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Left panel – dendrogram
            dendrogram(
                linkage_matrix,
                labels=[f"Structure {i}" for i in range(n)],
                ax=ax1,
                color_threshold=optimal_radius,
            )
            ax1.axhline(
                y=optimal_radius,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Radius = {optimal_radius:.4f}",
            )
            ax1.set_title("Hierarchical Clustering Dendrogram (PCA L2)")
            ax1.set_xlabel("Structure Index")
            ax1.set_ylabel("L2 Distance in PCA Space")
            ax1.legend()

            # Right panel – threshold vs cluster count
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
                x=optimal_radius,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Radius = {optimal_radius:.4f}",
            )
            ax2.scatter(
                [optimal_radius],
                [threshold_clustering["n_clusters"]],
                color="red",
                s=100,
                zorder=5,
                label=f"Selected ({threshold_clustering['n_clusters']} clusters)",
            )

            ax2.set_xlabel("L2 Distance in PCA Space")
            ax2.set_ylabel("Number of Clusters")
            ax2.set_title("Radius vs Cluster Count with Exponential Decay Fit")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(
                "approximate_clustering_analysis.png", dpi=300, bbox_inches="tight"
            )
            print(
                "Approximate clustering plots saved to approximate_clustering_analysis.png"
            )

            try:
                plt.show()
            except Exception:
                print("Note: Could not display plot interactively, but saved to file")
        except ImportError:
            print(
                "Warning: matplotlib not available, skipping visualization",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 6. Summary output (mirrors exact mode)
    # ------------------------------------------------------------------
    print(f"\nFound {len(all_threshold_data)} different clustering configurations")
    print(
        f"Radius range: {all_threshold_data[0]['nrmsd_threshold']:.6f} "
        f"to {all_threshold_data[-1]['nrmsd_threshold']:.6f}"
    )
    print(
        f"Cluster count range: {len(all_threshold_data[-1]['clusters'])} "
        f"to {len(all_threshold_data[0]['clusters'])}"
    )

    print(f"\nClustering at auto-detected radius {optimal_radius:.6f}:")
    print(f"  Number of clusters: {threshold_clustering['n_clusters']}")
    print(f"  Cluster sizes: {threshold_clustering['cluster_sizes']}")
    for i, cluster in enumerate(threshold_clustering["clusters"]):
        print(
            f"  Cluster {i + 1}: {cluster['representative']} "
            f"+ {len(cluster['members'])} members"
        )

    if output_json:
        output_data = {
            "all_thresholds": all_threshold_data,
            "selected_clustering": threshold_clustering,
            "parameters": {
                "mode": "approximate",
                "radius": optimal_radius,
                "radius_source": "auto-detected",
                "n_structures": len(structures),
            },
        }
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nApproximate clustering results saved to {output_json}")


# ----------------------------------------------------------------------


def find_all_thresholds_and_clusters(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray, file_paths: List[Path]
) -> List[dict]:
    """Enumerate all merge distances and build clustering summaries for each threshold.

    Args:
        distance_matrix (np.ndarray): Square nRMSD distance matrix.
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix.
        file_paths (List[Path]): Paths corresponding to the structures.

    Returns:
        List of clustering descriptions for each threshold value.
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
    """Compute pairwise nRMSD distance matrix for a set of structures.

    Distances are cached on disk to avoid recomputation between runs.

    Args:
        structures (List[Structure]): Structures to compare.
        file_paths (List[Path]): Paths corresponding to the structures.
        cache (NRMSDCache): Cache object storing previously computed nRMSD values.
        visualize (bool): Unused here, kept for interface compatibility.
        rmsd_method (str): RMSD method name ("quaternions", "svd", "qcp", "validate").

    Returns:
        np.ndarray: Square matrix of nRMSD distances between all structures.
    """
    n_structures = len(structures)

    if n_structures == 1:
        return np.zeros((1, 1))

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

    # Return distance matrix for further processing
    return distance_matrix


def get_clustering_at_threshold(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    threshold: float,
) -> dict:
    """Compute clustering summary for a chosen nRMSD threshold.

    Args:
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix.
        distance_matrix (np.ndarray): Square nRMSD distance matrix.
        file_paths (List[Path]): Paths corresponding to the structures.
        threshold (float): nRMSD cut-off defining clusters.

    Returns:
        dict: Dictionary with cluster counts, sizes and representatives.
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
    """Evaluate an exponential decay model y = a * exp(-b * x) + c.

    Args:
        x (np.ndarray): Input x-coordinates.
        a (float): Amplitude parameter.
        b (float): Decay rate parameter.
        c (float): Offset parameter.

    Returns:
        np.ndarray: Model values for each x.
    """
    return a * np.exp(-b * x) + c


def fit_exponential_decay(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an exponential decay to threshold–cluster data and locate key points.

    The fit is used to generate a smooth curve and identify "knee"-like
    inflection candidates in the decay.

    Args:
        x (np.ndarray): Threshold values.
        y (np.ndarray): Cluster counts for each threshold.

    Returns:
        tuple:
            A tuple ``(x_smooth, y_smooth, inflection_x)`` containing:

            - **x_smooth**: smooth x grid for plotting the fitted curve.
            - **y_smooth**: model values evaluated on ``x_smooth``.
            - **inflection_x**: one or more x positions of key curvature points.
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
    """Automatically choose an nRMSD threshold from the decay of cluster counts.

    The method computes cluster counts over all merge distances and uses the
    fitted exponential decay to identify a suitable "knee" as optimal threshold.

    Args:
        distance_matrix (np.ndarray): Square nRMSD distance matrix (unused here).
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix.

    Returns:
        float: Selected threshold value (or a fallback default).
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
    """Find medoid indices (best representatives) for each cluster.

    Args:
        clusters (List[List[int]]): Cluster membership as lists of indices.
        distance_matrix (np.ndarray): Square nRMSD distance matrix.

    Returns:
        Index of the medoid structure for each cluster.
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
    """Entry point for the distiller CLI: parse args, load structures and run clustering."""
    args = parse_arguments()

    # Combine file paths from CLI arguments and/or stdin
    file_paths: List[Path] = []
    cli_paths = [p for p in args.files if str(p) != "-"]
    file_paths.extend(cli_paths)

    # If no CLI paths provided or '-' sentinel present, read from stdin
    if not args.files or any(str(p) == "-" for p in args.files):
        stdin_paths = [Path(line.strip()) for line in sys.stdin if line.strip()]
        file_paths.extend(stdin_paths)

    # Validate input files
    valid_files = validate_input_files(file_paths)

    if not valid_files:
        print("Error: No valid input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(valid_files)} files")

    # Parse all structure files
    print("Parsing structure files...")
    structures: List[Structure] = []
    parsed_files: List[Path] = []

    for file_path in tqdm(valid_files, desc="Parsing", unit="file"):
        try:
            structure = parse_structure_file(file_path)
            structures.append(structure)
            parsed_files.append(file_path)
        except Exception:
            # Keep reporting failures explicitly
            print(f"  Failed to parse {file_path}, skipping", file=sys.stderr)

    # Replace the original list with the successfully parsed ones
    valid_files = parsed_files

    if not structures:
        print("Error: No structures could be parsed", file=sys.stderr)
        sys.exit(1)

    # valid_files already filtered to successfully parsed structures above

    # Validate nucleotide counts
    print("\nValidating nucleotide counts...")
    validate_nucleotide_counts(structures, valid_files)

    # Switch workflow based on requested mode
    if args.mode == "approximate":
        run_approximate_multiple(
            structures,
            valid_files,
            args.radius,
            args.output_json,
            visualize=args.visualize,
            method=args.method,
            preference=args.preference,
            damping=args.damping,
            n_representatives=args.n_representatives,
        )
        return
    else:
        run_exact(structures, valid_files, args)
        return


if __name__ == "__main__":
    main()
