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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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
                          or auto-detected N maximally representative
                          structures.
  radius-graph           Large-dataset graph clustering based on connected
                          components in a kNN graph pruned by radius.

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

  # Auto-detect representative count via facility location
  distiller --method facility-location *.cif

  # Select exactly 5 representatives via facility location
  distiller --method facility-location --n-representatives 5 *.cif

  # Fast approximate facility location for large datasets
  distiller --method facility-location --approx-backend graph *.cif

  # Use FAISS for graph neighbor search when installed
  distiller --method radius-graph --approx-backend graph --neighbor-search faiss *.cif

  # Fast approximate clustering for large datasets
  distiller --method radius-graph *.cif

  # Approximate mode with an explicit radius
  distiller --radius 1.0 *.cif

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
            "'approximate' uses PCA + reduced-space L2 distances; "
            "'exact' uses rigorous pairwise nRMSD"
        ),
    )

    mode_group.add_argument(
        "--method",
        choices=[
            "hierarchical",
            "affinity-propagation",
            "facility-location",
            "radius-graph",
        ],
        default="hierarchical",
        help=("clustering algorithm (default: hierarchical)"),
    )

    mode_group.add_argument(
        "--n-representatives",
        type=int,
        default=None,
        metavar="N",
        help=(
            "number of representatives for --method facility-location; "
            "if omitted, auto-detected from the exponential-decay knee"
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
        default=None,
        help=(
            "PCA-space radius for approximate hierarchical or radius-graph "
            "clustering. if omitted, auto-detected from the exponential-decay knee"
        ),
    )

    # -- Approximate backend options --------------------------------------
    approx_group = parser.add_argument_group("approximate mode backend options")

    approx_group.add_argument(
        "--approx-backend",
        choices=["dense", "graph"],
        default="dense",
        help=(
            "approximate-mode backend: 'dense' builds the full PCA-space "
            "distance matrix; 'graph' uses a sparse kNN graph for large datasets"
        ),
    )

    approx_group.add_argument(
        "--n-neighbors",
        type=int,
        default=32,
        metavar="K",
        help=(
            "number of nearest neighbors to retain in graph backend "
            "(default: 32)"
        ),
    )

    approx_group.add_argument(
        "--neighbor-search",
        choices=["sklearn", "faiss"],
        default="sklearn",
        help=(
            "graph-backend neighbor search engine. 'faiss' is optional and "
            "falls back to an error if not installed"
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


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Reject mutually incompatible CLI argument combinations."""
    if args.n_neighbors < 1:
        print("Error: --n-neighbors must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.n_representatives is not None and args.method != "facility-location":
        print(
            "Error: --n-representatives requires --method facility-location",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode != "exact" and args.threshold is not None:
        print("Error: --threshold is only valid in --mode exact", file=sys.stderr)
        sys.exit(1)

    if args.mode != "approximate" and args.radius is not None:
        print("Error: --radius is only valid in --mode approximate", file=sys.stderr)
        sys.exit(1)

    if args.mode != "approximate" and args.approx_backend != "dense":
        print(
            "Error: --approx-backend is only valid in --mode approximate",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode != "approximate" and args.n_neighbors != 32:
        print(
            "Error: --n-neighbors is only valid in --mode approximate",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode != "approximate" and args.neighbor_search != "sklearn":
        print(
            "Error: --neighbor-search is only valid in --mode approximate",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.method == "radius-graph" and args.mode != "approximate":
        print(
            "Error: --method radius-graph is only valid in --mode approximate",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.method != "hierarchical" and args.threshold is not None:
        print(
            "Error: --threshold is only valid with --method hierarchical",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.method not in {"hierarchical", "radius-graph"} and args.radius is not None:
        print(
            "Error: --radius is only valid with --method hierarchical or radius-graph",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.method != "affinity-propagation" and args.preference is not None:
        print(
            "Error: --preference is only valid with --method affinity-propagation",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.approx_backend == "graph" and args.method in {
        "hierarchical",
        "affinity-propagation",
    }:
        print(
            "Error: graph backend supports --method facility-location or radius-graph",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.approx_backend != "graph" and args.neighbor_search != "sklearn":
        print(
            "Error: --neighbor-search is only valid with --approx-backend graph",
            file=sys.stderr,
        )
        sys.exit(1)


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
) -> Tuple[dict, dict]:
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
        Tuple of clustering result dict and method diagnostics.
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

    clustering = build_clustering_from_groups(
        [clusters.get(cluster_id, []) for cluster_id in range(n_clusters)],
        distance_matrix,
        file_paths,
        representative_indices={
            cluster_id: int(exemplar_indices[cluster_id]) for cluster_id in range(n_clusters)
        },
    )
    diagnostics = {
        "exemplar_indices": [int(idx) for idx in exemplar_indices],
        "labels": [int(label) for label in labels],
        "preference": None if preference is None else float(preference),
        "damping": float(damping),
    }
    return clustering, diagnostics


def cluster_facility_location(
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    n_representatives: Optional[int] = None,
) -> Tuple[dict, dict, dict]:
    """Select *n_representatives* using submodular facility location.

    Uses the *apricot-select* library to greedily pick a maximally
    representative subset.  The distance matrix is converted to a
    non-negative similarity matrix via ``max(d) - d``.

    Args:
        distance_matrix: Square symmetric distance matrix.
        file_paths: Paths corresponding to the structures.
        n_representatives: Exact number of representatives to select.
            ``None`` triggers auto-detection from the facility-location gain curve.

    Returns:
        Tuple of clustering result dict, selection metadata and diagnostics.
    """
    try:
        from apricot import FacilityLocationSelection
    except ImportError:
        print(
            "Error: apricot-select is required for facility-location clustering. "
            "Install it with:  pip install apricot-select",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(file_paths)
    if n == 0:
        print("Error: No structures available for facility location", file=sys.stderr)
        sys.exit(1)

    if n_representatives is not None and n_representatives < 1:
        print(
            "Error: --n-representatives must be at least 1",
            file=sys.stderr,
        )
        sys.exit(1)

    requested_n = n if n_representatives is None else n_representatives
    if requested_n >= n:
        print(
            f"Warning: requested representative count ({requested_n}) >= number of "
            f"structures ({n}); selecting all structures.",
            file=sys.stderr,
        )
        requested_n = n

    # apricot requires a non-negative similarity matrix
    similarity = distance_matrix.max() - distance_matrix
    np.fill_diagonal(similarity, 0.0)

    selector = FacilityLocationSelection(
        n_samples=requested_n,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(similarity)

    full_ranking = [int(r) for r in selector.ranking]
    full_gains = [float(g) for g in selector.gains]

    if n_representatives is None:
        auto_n = determine_optimal_representative_count(full_gains)
        selection = {
            "variable": "n_representatives",
            "value": auto_n,
            "unit": "count",
            "source": "auto-detected",
            "rule": "exponential-decay-knee",
        }
    else:
        auto_n = requested_n
        selection = {
            "variable": "n_representatives",
            "value": auto_n,
            "unit": "count",
            "source": "user-specified",
            "rule": None,
        }

    ranking = full_ranking[:auto_n]
    gains = full_gains[:auto_n]

    print(
        f"Facility location selected {auto_n} representatives "
        f"(total gain: {sum(gains):.4f})"
    )

    groups = assign_to_representatives(distance_matrix, ranking)
    clustering = build_clustering_from_representatives(groups, file_paths)
    diagnostics = {
        "selection_curve": build_facility_location_selection_curve(full_gains),
        "ranking": ranking,
        "gains": gains,
        "full_ranking": full_ranking,
        "full_gains": full_gains,
    }
    return clustering, selection, diagnostics


def cluster_facility_location_embedding(
    embedding: np.ndarray,
    file_paths: List[Path],
    n_neighbors: int,
    neighbor_search: str = "sklearn",
    n_representatives: Optional[int] = None,
) -> Tuple[dict, dict, dict]:
    """Select representatives directly from PCA embeddings using sparse neighbors."""
    try:
        from apricot import FacilityLocationSelection
    except ImportError:
        print(
            "Error: apricot-select is required for facility-location clustering. "
            "Install it with:  pip install apricot-select",
            file=sys.stderr,
        )
        sys.exit(1)

    n = len(file_paths)
    if n == 0:
        print("Error: No structures available for facility location", file=sys.stderr)
        sys.exit(1)

    if n_representatives is not None and n_representatives < 1:
        print("Error: --n-representatives must be at least 1", file=sys.stderr)
        sys.exit(1)

    requested_n = n if n_representatives is None else min(n_representatives, n)
    selector = FacilityLocationSelection(
        n_samples=requested_n,
        metric="euclidean",
        optimizer="lazy",
        n_neighbors=min(max(1, n_neighbors), n),
        verbose=False,
    )
    selector.fit(embedding)

    full_ranking = [int(r) for r in selector.ranking]
    full_gains = [float(g) for g in selector.gains]

    if n_representatives is None:
        selected_n = determine_optimal_representative_count(full_gains)
        selection = {
            "variable": "n_representatives",
            "value": selected_n,
            "unit": "count",
            "source": "auto-detected",
            "rule": "exponential-decay-knee",
        }
    else:
        selected_n = requested_n
        selection = {
            "variable": "n_representatives",
            "value": selected_n,
            "unit": "count",
            "source": "user-specified",
            "rule": None,
        }

    ranking = full_ranking[:selected_n]
    gains = full_gains[:selected_n]
    groups = assign_embedding_to_representatives(embedding, ranking)
    clustering = build_clustering_from_representatives(groups, file_paths)
    diagnostics = {
        "selection_curve": build_facility_location_selection_curve(full_gains),
        "ranking": ranking,
        "gains": gains,
        "full_ranking": full_ranking,
        "full_gains": full_gains,
        "n_neighbors": min(max(1, n_neighbors), n),
        "approximate_search": True,
        "neighbor_search": neighbor_search,
    }
    print(
        f"Facility location selected {selected_n} representatives "
        f"(total gain: {sum(gains):.4f})"
    )
    return clustering, selection, diagnostics


def assign_embedding_to_representatives(
    embedding: np.ndarray, representative_indices: List[int]
) -> Dict[int, List[int]]:
    """Assign each embedding vector to its nearest representative in embedding space."""
    representative_array = np.asarray(representative_indices, dtype=int)
    groups = {int(idx): [int(idx)] for idx in representative_indices}

    if len(representative_array) == 0:
        return groups

    representative_vectors = embedding[representative_array]
    for idx in range(embedding.shape[0]):
        if idx in groups:
            continue
        diffs = representative_vectors - embedding[idx]
        best = int(representative_array[int(np.argmin(np.sum(diffs * diffs, axis=1)))])
        groups[best].append(idx)

    return groups


def build_knn_graph(
    embedding: np.ndarray,
    n_neighbors: int,
    neighbor_search: str = "sklearn",
) -> Tuple[List[set[int]], List[Tuple[int, int, float]], dict]:
    """Build a symmetric kNN graph in PCA space."""
    n = embedding.shape[0]
    if n == 0:
        return [], [], {
            "n_neighbors": 0,
            "n_edges": 0,
            "avg_degree": 0.0,
            "neighbor_search": neighbor_search,
        }

    k = min(max(1, n_neighbors), n)
    if neighbor_search == "faiss":
        distances, indices = query_knn_faiss(embedding, k)
    else:
        model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        model.fit(embedding)
        distances, indices = model.kneighbors(embedding)

    adjacency = [set() for _ in range(n)]
    edge_weights: dict[Tuple[int, int], float] = {}

    for source in range(n):
        for neighbor_idx, dist in zip(indices[source], distances[source]):
            target = int(neighbor_idx)
            if source == target:
                continue
            edge = (source, target) if source < target else (target, source)
            previous = edge_weights.get(edge)
            value = float(dist)
            if previous is None or value < previous:
                edge_weights[edge] = value
            adjacency[source].add(target)
            adjacency[target].add(source)

    edges = [(left, right, dist) for (left, right), dist in edge_weights.items()]
    avg_degree = float(sum(len(neighbors) for neighbors in adjacency) / n)
    diagnostics = {
        "n_neighbors": k,
        "n_edges": len(edges),
        "avg_degree": avg_degree,
        "neighbor_search": neighbor_search,
    }
    return adjacency, edges, diagnostics


def query_knn_faiss(embedding: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Query k-nearest neighbors with FAISS if available."""
    try:
        import faiss
    except ImportError:
        print(
            "Error: --neighbor-search faiss requires faiss-cpu. "
            "Install it with: pip install faiss-cpu",
            file=sys.stderr,
        )
        sys.exit(1)

    vectors = np.ascontiguousarray(embedding.astype(np.float32))
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    squared_distances, indices = index.search(vectors, n_neighbors)
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    return distances, indices


def connected_components_from_adjacency(adjacency: List[set[int]]) -> List[List[int]]:
    """Compute connected components of an adjacency list."""
    components: List[List[int]] = []
    visited = [False] * len(adjacency)

    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))

    return components


def build_radius_graph_candidates(
    embedding: np.ndarray,
    file_paths: List[Path],
    n_neighbors: int,
    neighbor_search: str = "sklearn",
) -> Tuple[List[dict], dict]:
    """Build radius-graph clustering candidates from kNN edge distances."""
    adjacency, edges, graph_stats = build_knn_graph(
        embedding, n_neighbors, neighbor_search=neighbor_search
    )
    if not edges:
        clustering = build_clustering_from_groups(
            [[idx] for idx in range(len(file_paths))],
            pairwise_squared_l2(embedding),
            file_paths,
        )
        return [{"value": 0.0, **clustering}], graph_stats

    distance_matrix = pairwise_squared_l2(embedding)
    candidate_radii = sorted({float(dist) for _, _, dist in edges})
    candidates: List[dict] = []

    for radius in candidate_radii:
        pruned = [
            {
                neighbor
                for neighbor in neighbors
                if distance_matrix[idx, neighbor] <= radius + 1e-12
            }
            for idx, neighbors in enumerate(adjacency)
        ]
        clustering = build_clustering_from_groups(
            connected_components_from_adjacency(pruned), distance_matrix, file_paths
        )
        candidates.append({"value": float(radius), **clustering})

    return candidates, graph_stats


def pairwise_squared_l2(embedding: np.ndarray) -> np.ndarray:
    """Compute a dense pairwise Euclidean distance matrix from embeddings."""
    if embedding.shape[0] <= 1:
        return np.zeros((embedding.shape[0], embedding.shape[0]), dtype=float)
    return squareform(pdist(embedding.astype(np.float64), metric="euclidean"))


def run_radius_graph_workflow(
    *,
    embedding: np.ndarray,
    file_paths: List[Path],
    radius: Optional[float],
    n_neighbors: int,
    neighbor_search: str,
    output_json: Optional[str],
    visualize: bool,
    extra_parameters: Optional[dict] = None,
) -> None:
    """Run scalable radius-graph clustering in PCA space."""
    extra_parameters = extra_parameters or {}
    candidates, graph_stats = build_radius_graph_candidates(
        embedding, file_paths, n_neighbors, neighbor_search=neighbor_search
    )
    candidate_values = np.array([entry["value"] for entry in candidates], dtype=float)
    candidate_counts = np.array([entry["n_clusters"] for entry in candidates], dtype=float)

    if radius is None:
        selected_radius = determine_optimal_decay_value(
            candidate_values,
            candidate_counts,
            fallback=float(candidate_values[len(candidate_values) // 2])
            if len(candidate_values)
            else 0.0,
        )
        selection_source = "auto-detected"
    else:
        selected_radius = float(radius)
        selection_source = "user-specified"
        print(f"Using user-specified radius: {selected_radius}")

    selected_clustering = min(
        candidates,
        key=lambda entry: (abs(entry["value"] - selected_radius), entry["value"]),
    )
    summarize_hierarchical_candidates(
        "Radius",
        candidates,
        selected_clustering,
        selected_clustering["value"],
    )

    diagnostics = build_hierarchical_diagnostics(
        candidates,
        x_label="pca-l2",
        y_label="number_of_clusters",
    )
    diagnostics.update(graph_stats)
    diagnostics["approximate_search"] = True

    if output_json:
        _write_json(
            output_json,
            build_output_data(
                mode="approximate",
                method="radius-graph",
                distance_metric="pca-l2",
                n_structures=len(file_paths),
                clustering={
                    "n_clusters": selected_clustering["n_clusters"],
                    "cluster_sizes": selected_clustering["cluster_sizes"],
                    "clusters": selected_clustering["clusters"],
                },
                selection=build_hierarchical_selection(
                    value=selected_clustering["value"],
                    unit="pca-l2",
                    source=selection_source,
                ),
                diagnostics=diagnostics,
                parameters={
                    **extra_parameters,
                    "approx_backend": "graph",
                    "n_neighbors": graph_stats["n_neighbors"],
                },
            ),
        )

    if visualize:
        print(
            "Warning: visualization is not supported for --method radius-graph; skipping",
            file=sys.stderr,
        )


def assign_to_representatives(
    distance_matrix: np.ndarray, representative_indices: List[int]
) -> Dict[int, List[int]]:
    """Assign each structure to its nearest representative."""
    rep_indices = np.array([int(idx) for idx in representative_indices], dtype=int)
    selected_set = set(int(idx) for idx in representative_indices)
    groups = {int(idx): [int(idx)] for idx in representative_indices}

    for idx in range(distance_matrix.shape[0]):
        if idx in selected_set:
            continue
        dists_to_reps = distance_matrix[idx, rep_indices]
        best = int(rep_indices[int(np.argmin(dists_to_reps))])
        groups[best].append(idx)

    return groups


def build_clustering_from_representatives(
    groups: Dict[int, List[int]], file_paths: List[Path]
) -> dict:
    """Build a clustering result from representative-indexed groups."""
    cluster_records = []
    for representative_idx, member_indices in groups.items():
        members = [
            str(file_paths[idx]) for idx in member_indices if idx != representative_idx
        ]
        cluster_records.append(
            {
                "representative": str(file_paths[representative_idx]),
                "members": members,
            }
        )

    cluster_records.sort(
        key=lambda record: (-1 - len(record["members"]), record["representative"])
    )
    return {
        "n_clusters": len(cluster_records),
        "cluster_sizes": [1 + len(record["members"]) for record in cluster_records],
        "clusters": cluster_records,
    }


def build_clustering_from_groups(
    groups: List[List[int]],
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    representative_indices: Optional[Dict[int, int]] = None,
) -> dict:
    """Build a clustering result from groups of structure indices."""
    cluster_records = []

    for group_id, group in enumerate(groups):
        if not group:
            continue
        if representative_indices is None:
            representative_idx = find_cluster_medoids([group], distance_matrix)[0]
        else:
            representative_idx = representative_indices[group_id]
        members = [str(file_paths[idx]) for idx in group if idx != representative_idx]
        cluster_records.append(
            {
                "representative": str(file_paths[representative_idx]),
                "members": members,
            }
        )

    cluster_records.sort(
        key=lambda record: (-1 - len(record["members"]), record["representative"])
    )
    return {
        "n_clusters": len(cluster_records),
        "cluster_sizes": [1 + len(record["members"]) for record in cluster_records],
        "clusters": cluster_records,
    }


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
    output_data: dict,
) -> None:
    """Write unified clustering results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nClustering results saved to {output_path}")


def build_output_data(
    *,
    mode: str,
    method: str,
    distance_metric: str,
    n_structures: int,
    clustering: dict,
    selection: dict,
    diagnostics: Optional[dict] = None,
    parameters: Optional[dict] = None,
) -> dict:
    """Build the unified JSON output payload."""
    return {
        "parameters": {
            "mode": mode,
            "method": method,
            "distance_metric": distance_metric,
            "n_structures": n_structures,
            **(parameters or {}),
        },
        "selection": selection,
        "clustering": clustering,
        "diagnostics": diagnostics or {},
    }


def build_hierarchical_selection(
    *,
    value: float,
    unit: str,
    source: str,
) -> dict:
    """Build selection metadata for hierarchical clustering."""
    return {
        "variable": "distance_cutoff",
        "value": float(value),
        "unit": unit,
        "source": source,
        "rule": "exponential-decay-knee" if source == "auto-detected" else None,
    }


def build_hierarchical_diagnostics(
    candidates: List[dict], x_label: str, y_label: str
) -> dict:
    """Build diagnostics shared by hierarchical clustering modes."""
    if not candidates:
        return {"selection_curve": [], "axes": {"x": x_label, "y": y_label}}

    x = np.array([entry["value"] for entry in candidates], dtype=float)
    y = np.array([entry["n_clusters"] for entry in candidates], dtype=float)
    x_smooth, y_smooth, knee_x = fit_exponential_decay(x, y)

    return {
        "selection_curve": candidates,
        "axes": {"x": x_label, "y": y_label},
        "fit": {
            "x": [float(value) for value in x_smooth],
            "y": [float(value) for value in y_smooth],
            "knee_candidates": [float(value) for value in knee_x],
        },
    }


def build_facility_location_selection_curve(gains: List[float]) -> List[dict]:
    """Build the facility-location decay curve from cumulative gains."""
    if not gains:
        return []

    gains_array = np.asarray(gains, dtype=float)
    remaining_gain = gains_array.sum() - np.cumsum(gains_array)
    return [
        {
            "value": int(index + 1),
            "n_clusters": int(index + 1),
            "marginal_gain": float(gain),
            "remaining_gain": float(remaining),
        }
        for index, (gain, remaining) in enumerate(zip(gains_array, remaining_gain))
    ]


def determine_optimal_representative_count(gains: List[float]) -> int:
    """Auto-detect the representative count from facility-location gains."""
    if not gains:
        return 1

    curve = build_facility_location_selection_curve(gains)
    x = np.array([entry["value"] for entry in curve], dtype=float)
    y = np.array([entry["remaining_gain"] for entry in curve], dtype=float)
    optimal_value = determine_optimal_decay_value(x, y, fallback=float(len(gains)))
    optimal_n = int(round(optimal_value))
    optimal_n = max(1, min(len(gains), optimal_n))
    print(f"Auto-detected representative count: {optimal_n}")
    return optimal_n


def determine_optimal_decay_value(
    x: np.ndarray, y: np.ndarray, fallback: float
) -> float:
    """Choose a knee-like value from a decay curve."""
    _, _, inflection_x = fit_exponential_decay(x, y)

    if len(inflection_x) > 0:
        optimal_value = float(inflection_x[0])
        print(f"Auto-detected optimal value: {optimal_value:.6f}")
        return optimal_value

    print(f"No inflection points found, using fallback value: {fallback}")
    return float(fallback)


def summarize_hierarchical_candidates(
    label: str, candidates: List[dict], selected_clustering: dict, selected_value: float
) -> None:
    """Print a human-readable summary of hierarchical clustering candidates."""
    print(f"\nFound {len(candidates)} different clustering configurations")
    print(
        f"{label} range: {candidates[0]['value']:.6f} to {candidates[-1]['value']:.6f}"
    )
    print(
        f"Cluster count range: {candidates[-1]['n_clusters']} to {candidates[0]['n_clusters']}"
    )

    print(f"\nClustering at {label.lower()} {selected_value:.6f}:")
    _print_clustering_summary(selected_clustering)


def visualize_hierarchical_clustering(
    *,
    linkage_matrix: np.ndarray,
    candidates: List[dict],
    selected_value: float,
    selected_clustering: dict,
    dendrogram_title: str,
    x_label: str,
    y_label: str,
    selected_label: str,
    output_file: str,
) -> None:
    """Render the standard hierarchical dendrogram + decay plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    dendrogram(
        linkage_matrix,
        labels=[f"Structure {i}" for i in range(linkage_matrix.shape[0] + 1)],
        ax=ax1,
        color_threshold=selected_value,
    )
    ax1.axhline(
        y=selected_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{selected_label} = {selected_value:.6f}",
    )
    ax1.set_title(dendrogram_title)
    ax1.set_xlabel("Structure Index")
    ax1.set_ylabel(y_label)
    ax1.legend()

    x = np.array([entry["value"] for entry in candidates], dtype=float)
    y = np.array([entry["n_clusters"] for entry in candidates], dtype=float)
    x_smooth, y_smooth, inflection_x = fit_exponential_decay(x, y)

    ax2.scatter(x, y, alpha=0.7, s=30, label="Data points")
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
        x=selected_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"{selected_label} = {selected_value:.6f}",
    )
    ax2.scatter(
        [selected_value],
        [selected_clustering["n_clusters"]],
        color="red",
        s=100,
        zorder=5,
        label=f"Selected ({selected_clustering['n_clusters']} clusters)",
    )
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title(f"{selected_label} vs Cluster Count with Exponential Decay Fit")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Clustering analysis plots saved to {output_file}")

    try:
        plt.show()
    except Exception:
        print("Note: Could not display plot interactively, but saved to file")


def run_distance_matrix_workflow(
    *,
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    mode: str,
    distance_metric: str,
    hierarchical_value: Optional[float],
    hierarchical_value_name: str,
    hierarchical_value_unit: str,
    preference: Optional[float],
    damping: float,
    n_representatives: Optional[int],
    method: str,
    visualize: bool,
    output_json: Optional[str],
    extra_parameters: Optional[dict] = None,
) -> None:
    """Run the selected clustering method on a prepared distance matrix."""
    extra_parameters = extra_parameters or {}

    if method == "facility-location":
        clustering, selection, diagnostics = cluster_facility_location(
            distance_matrix, file_paths, n_representatives
        )
        _print_clustering_summary(clustering)

        if visualize:
            labels = _labels_from_clustering(clustering, file_paths)
            exemplar_indices = np.array(diagnostics["ranking"], dtype=int)
            _visualize_flat_clustering(
                distance_matrix,
                labels,
                exemplar_indices,
                f"Facility Location Selection ({distance_metric})",
                "facility_location_analysis.png",
                gains=diagnostics.get("gains"),
            )

        if output_json:
            _write_json(
                output_json,
                build_output_data(
                    mode=mode,
                    method=method,
                    distance_metric=distance_metric,
                    n_structures=len(file_paths),
                    clustering=clustering,
                    selection=selection,
                    diagnostics=diagnostics,
                    parameters=extra_parameters,
                ),
            )
        return

    if method == "affinity-propagation":
        clustering, diagnostics = cluster_affinity_propagation(
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
                f"Affinity Propagation Clustering ({distance_metric})",
                "clustering_analysis.png",
            )

        if output_json:
            _write_json(
                output_json,
                build_output_data(
                    mode=mode,
                    method=method,
                    distance_metric=distance_metric,
                    n_structures=len(file_paths),
                    clustering=clustering,
                    selection={
                        "variable": "n_clusters",
                        "value": clustering["n_clusters"],
                        "unit": "count",
                        "source": "algorithm",
                        "rule": None,
                    },
                    diagnostics=diagnostics,
                    parameters={
                        **extra_parameters,
                        "preference": None if preference is None else float(preference),
                        "damping": float(damping),
                    },
                ),
            )
        return

    linkage_matrix = linkage(squareform(distance_matrix), method="complete")
    candidates = find_all_cutoffs_and_clusters(distance_matrix, linkage_matrix, file_paths)

    if hierarchical_value is None:
        selected_value = determine_optimal_cutoff(linkage_matrix)
        selection_source = "auto-detected"
    else:
        selected_value = hierarchical_value
        selection_source = "user-specified"
        print(f"Using user-specified {hierarchical_value_name}: {selected_value}")

    selected_clustering = get_clustering_at_cutoff(
        linkage_matrix, distance_matrix, file_paths, selected_value
    )

    if visualize:
        visualize_hierarchical_clustering(
            linkage_matrix=linkage_matrix,
            candidates=candidates,
            selected_value=selected_value,
            selected_clustering=selected_clustering,
            dendrogram_title=f"Hierarchical Clustering Dendrogram ({distance_metric})",
            x_label=hierarchical_value_unit,
            y_label=hierarchical_value_unit,
            selected_label=hierarchical_value_name.capitalize(),
            output_file=(
                "approximate_clustering_analysis.png"
                if mode == "approximate"
                else "clustering_analysis.png"
            ),
        )

    summarize_hierarchical_candidates(
        hierarchical_value_name.capitalize(),
        candidates,
        selected_clustering,
        selected_value,
    )

    if output_json:
        _write_json(
            output_json,
            build_output_data(
                mode=mode,
                method=method,
                distance_metric=distance_metric,
                n_structures=len(file_paths),
                clustering=selected_clustering,
                selection=build_hierarchical_selection(
                    value=selected_value,
                    unit=hierarchical_value_unit,
                    source=selection_source,
                ),
                diagnostics=build_hierarchical_diagnostics(
                    candidates,
                    x_label=hierarchical_value_unit,
                    y_label="number_of_clusters",
                ),
                parameters=extra_parameters,
            ),
        )


# ----------------------------------------------------------------------


def run_exact(
    structures: List[Structure], valid_files: List[Path], args: argparse.Namespace
) -> None:
    """Run exact nRMSD-based clustering workflow."""
    print("\nInitializing nRMSD cache...")
    cache = NRMSDCache(args.cache_file, args.cache_save_interval)

    print("\nComputing distance matrix...")
    distance_matrix = find_structure_clusters(
        structures, valid_files, cache, args.visualize, args.rmsd_method
    )

    run_distance_matrix_workflow(
        distance_matrix=distance_matrix,
        file_paths=valid_files,
        mode="exact",
        distance_metric="nrmsd",
        hierarchical_value=args.threshold,
        hierarchical_value_name="threshold",
        hierarchical_value_unit="nrmsd",
        preference=args.preference,
        damping=args.damping,
        n_representatives=args.n_representatives,
        method=args.method,
        visualize=args.visualize,
        output_json=args.output_json,
        extra_parameters={"rmsd_method": args.rmsd_method},
    )


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


def run_approximate(
    structures: List[Structure], file_paths: List[Path], args: argparse.Namespace
) -> None:
    """Run approximate PCA-based clustering workflow."""
    print("\nRunning approximate mode (feature-based PCA)")
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

    pca_solver = "randomized" if args.approx_backend == "graph" else "full"
    pca = PCA(n_components=0.95, svd_solver=pca_solver, random_state=0)
    X_red = pca.fit_transform(X).astype(np.float32)
    print(
        f"PCA reduced to {X_red.shape[1]} dimensions "
        f"({pca.explained_variance_ratio_.sum():.4f} variance)"
    )

    extra_parameters = {
        "pca_dimensions": int(X_red.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "approx_backend": args.approx_backend,
    }

    if args.approx_backend == "graph":
        print(
            f"Building sparse kNN graph backend with {min(args.n_neighbors, len(structures))} neighbors "
            f"using {args.neighbor_search}"
        )

        if args.method == "radius-graph":
            run_radius_graph_workflow(
                embedding=X_red,
                file_paths=file_paths,
                radius=args.radius,
                n_neighbors=args.n_neighbors,
                neighbor_search=args.neighbor_search,
                output_json=args.output_json,
                visualize=args.visualize,
                extra_parameters=extra_parameters,
            )
            return

        if args.method == "facility-location":
            clustering, selection, diagnostics = cluster_facility_location_embedding(
                X_red,
                file_paths,
                args.n_neighbors,
                args.neighbor_search,
                args.n_representatives,
            )
            _print_clustering_summary(clustering)

            if args.output_json:
                _write_json(
                    args.output_json,
                    build_output_data(
                        mode="approximate",
                        method="facility-location",
                        distance_metric="pca-l2",
                        n_structures=len(file_paths),
                        clustering=clustering,
                        selection=selection,
                        diagnostics=diagnostics,
                parameters={
                    **extra_parameters,
                    "n_neighbors": diagnostics["n_neighbors"],
                    "neighbor_search": args.neighbor_search,
                },
            ),
        )
            if args.visualize:
                print(
                    "Warning: visualization is not supported for graph-backend facility-location; skipping",
                    file=sys.stderr,
                )
            return

        print(
            "Error: graph backend supports --method facility-location or radius-graph",
            file=sys.stderr,
        )
        sys.exit(1)

    distance_matrix = pairwise_squared_l2(X_red)
    print(f"Pairwise L2 distance matrix computed ({len(structures)}×{len(structures)})")

    run_distance_matrix_workflow(
        distance_matrix=distance_matrix,
        file_paths=file_paths,
        mode="approximate",
        distance_metric="pca-l2",
        hierarchical_value=args.radius,
        hierarchical_value_name="radius",
        hierarchical_value_unit="pca-l2",
        preference=args.preference,
        damping=args.damping,
        n_representatives=args.n_representatives,
        method=args.method,
        visualize=args.visualize,
        output_json=args.output_json,
        extra_parameters=extra_parameters,
    )


# ----------------------------------------------------------------------


def find_all_cutoffs_and_clusters(
    distance_matrix: np.ndarray, linkage_matrix: np.ndarray, file_paths: List[Path]
) -> List[dict]:
    """Enumerate all merge distances and build clustering summaries for each cutoff.

    Args:
        distance_matrix (np.ndarray): Square distance matrix.
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix.
        file_paths (List[Path]): Paths corresponding to the structures.

    Returns:
        List of clustering descriptions for each cutoff value.
    """
    print("Finding all cutoff values where cluster assignments change...")
    cutoffs = np.sort(linkage_matrix[:, 2])
    print(f"Testing {len(cutoffs)} cutoff values where clustering changes:")

    results = []
    for cutoff in cutoffs:
        clustering = get_clustering_at_cutoff(
            linkage_matrix, distance_matrix, file_paths, float(cutoff)
        )
        print(
            f"  Cutoff {cutoff:.6f}: {clustering['n_clusters']} clusters, "
            f"sizes: {clustering['cluster_sizes']}"
        )
        results.append({"value": float(cutoff), **clustering})

    return results


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


def get_clustering_at_cutoff(
    linkage_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    file_paths: List[Path],
    cutoff: float,
) -> dict:
    """Compute clustering summary for a chosen hierarchical cutoff.

    Args:
        linkage_matrix (np.ndarray): Hierarchical clustering linkage matrix.
        distance_matrix (np.ndarray): Square distance matrix.
        file_paths (List[Path]): Paths corresponding to the structures.
        cutoff (float): Distance cut-off defining clusters.

    Returns:
        dict: Dictionary with cluster counts, sizes and representatives.
    """
    labels = fcluster(linkage_matrix, cutoff, criterion="distance")
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return build_clustering_from_groups(
        list(clusters.values()), distance_matrix, file_paths
    )


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


def determine_optimal_cutoff(linkage_matrix: np.ndarray) -> float:
    """Automatically choose a hierarchical cutoff from cluster-count decay."""
    cutoffs = np.sort(linkage_matrix[:, 2])
    cluster_counts = []
    for cutoff in cutoffs:
        labels = fcluster(linkage_matrix, cutoff, criterion="distance")
        cluster_counts.append(len(np.unique(labels)))

    return determine_optimal_decay_value(
        np.asarray(cutoffs, dtype=float),
        np.asarray(cluster_counts, dtype=float),
        fallback=float(cutoffs[len(cutoffs) // 2]) if len(cutoffs) else 0.1,
    )


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
    validate_cli_arguments(args)

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
        run_approximate(structures, valid_files, args)
        return
    else:
        run_exact(structures, valid_files, args)
        return


if __name__ == "__main__":
    main()
