#!/usr/bin/env python3
"""Quickly group nucleic-acid chains by spatial proximity.

Parses a PDB or mmCIF file, extracts a single representative atom per
nucleotide (C1' by default), builds a KD-tree and uses a distance
threshold to decide which chains are in contact.  Chains that share at
least one atom-atom contact below the threshold are placed in the same
group via union-find.

This is intentionally a fast heuristic -- no Residue/Structure objects
are constructed, altloc filtering is skipped (duplicate points are
harmless), and only one atom per nucleotide is considered.
"""

import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import IO, Dict, List, Optional, Set, Union

import numpy as np
import orjson
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from rnapolis.common import DNA_NAMES, RNA_NAMES
from rnapolis.parser import is_cif
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.util import handle_input_file

DEFAULT_ATOM_NAME = "C1'"
DEFAULT_DISTANCE_THRESHOLD = 15.0
_NA_RESIDUE_NAMES = RNA_NAMES | DNA_NAMES


def _resolve_columns(fmt: str, columns: pd.Index) -> dict[str, str]:
    """Map semantic column names to actual DataFrame columns for the given format."""
    resolved: dict[str, str] = {}
    specs = {
        "atom_name": (("auth_atom_id", "label_atom_id") if fmt == "mmCIF" else "name"),
        "chain_id": (
            ("auth_asym_id", "label_asym_id") if fmt == "mmCIF" else "chainID"
        ),
        "x": "Cartn_x" if fmt == "mmCIF" else "x",
        "y": "Cartn_y" if fmt == "mmCIF" else "y",
        "z": "Cartn_z" if fmt == "mmCIF" else "z",
        "model": "pdbx_PDB_model_num" if fmt == "mmCIF" else "model",
        "residue_name": (
            ("auth_comp_id", "label_comp_id") if fmt == "mmCIF" else "resName"
        ),
    }
    for key, spec in specs.items():
        if isinstance(spec, tuple):
            for col in spec:
                if col in columns:
                    resolved[key] = col
                    break
        else:
            if spec in columns:
                resolved[key] = spec
    return resolved


def _extract_c1_atoms(
    atoms: pd.DataFrame,
    atom_name: str,
    model: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Filter to representative atoms in one model; return (chains, coords).

    Returns two arrays: ``chains`` (str, shape N) and ``coords``
    (float64, shape N×3).
    """
    if atoms.empty:
        return np.array([], dtype=object), np.empty((0, 3), dtype=float)

    fmt = atoms.attrs.get("format", "PDB")
    cols = _resolve_columns(fmt, atoms.columns)
    atom_col = cols.get("atom_name")
    chain_col = cols.get("chain_id")
    model_col = cols.get("model")
    if atom_col is None or chain_col is None:
        return np.array([], dtype=object), np.empty((0, 3), dtype=float)

    mask = atoms[atom_col].astype(str) == atom_name
    filtered = atoms[mask]

    if model_col and model_col in filtered.columns and not filtered.empty:
        if model is None:
            first_model = filtered[model_col].dropna().min()
            filtered = filtered[filtered[model_col] == first_model]
        else:
            filtered = filtered[filtered[model_col] == model]

    if filtered.empty:
        return np.array([], dtype=object), np.empty((0, 3), dtype=float)

    resname_col = cols.get("residue_name")
    if resname_col and resname_col in filtered.columns:
        chain_arr = filtered[chain_col].astype(str).to_numpy()
        resnames = filtered[resname_col].astype(str).str.upper().to_numpy()
        is_na = np.isin(resnames, list(_NA_RESIDUE_NAMES))
        na_counts: dict[str, int] = {}
        for chain, na in zip(chain_arr, is_na):
            na_counts[chain] = na_counts.get(chain, 0) + int(na)
        na_chains = {c for c, n in na_counts.items() if n > 0}
        keep_mask = np.isin(chain_arr, list(na_chains))
        filtered = filtered[keep_mask]

    if filtered.empty:
        return np.array([], dtype=object), np.empty((0, 3), dtype=float)

    chains = filtered[chain_col].astype(str).to_numpy()
    coords = filtered[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=float)

    return chains, coords


def _union_find(chains: np.ndarray, pairs: Set[tuple[int, int]]) -> List[Set[str]]:
    """Group chains into connected components using union-find."""
    unique_chains = set(chains.tolist())
    parent: dict[str, str] = {c: c for c in unique_chains}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        ci, cj = chains[i], chains[j]
        if ci != cj:
            union(ci, cj)

    groups: dict[str, Set[str]] = {}
    for c in unique_chains:
        root = find(c)
        groups.setdefault(root, set()).add(c)

    return sorted(groups.values(), key=lambda g: sorted(g))


def find_na_chain_groups(
    content: Union[str, IO[str]],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    atom_name: str = DEFAULT_ATOM_NAME,
    model: Optional[int] = None,
) -> List[Set[str]]:
    """Find groups of nucleic-acid chains in contact within a structure.

    Parameters
    ----------
    content:
        PDB or mmCIF content as a string or file-like object.
    distance_threshold:
        Maximum distance (in Angstroms) between two representative atoms
        for their chains to be considered in contact.
    atom_name:
        Atom name to use as representative (default ``"C1'"``).  Must be
        a sugar/base atom present once per nucleotide and absent from
        non-nucleic-acid residues.
    model:
        Model number to analyse.  If ``None``, the first model
        containing the representative atom is used.

    Returns
    -------
    list[set[str]]
        Sorted list of chain groups.  Each group is a set of chain IDs
        that are in spatial contact.  Isolated chains appear as
        singletons.
    """
    if isinstance(content, str):
        content_str = content
    else:
        content.seek(0)
        content_str = content.read()
        if isinstance(content_str, bytes):
            content_str = content_str.decode("utf-8")

    format_is_cif = is_cif(io.StringIO(content_str))
    atoms = (
        parse_cif_atoms(content_str) if format_is_cif else parse_pdb_atoms(content_str)
    )

    chains, coords = _extract_c1_atoms(atoms, atom_name, model)
    if len(chains) == 0:
        return []
    if len(chains) == 1:
        return [{str(chains[0])}]

    tree = KDTree(coords)
    pairs = tree.query_pairs(distance_threshold)

    return _union_find(chains, pairs)


def find_na_chain_groups_file(
    path: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    atom_name: str = DEFAULT_ATOM_NAME,
    model: Optional[int] = None,
) -> List[Set[str]]:
    """Read a file and find nucleic-acid chain groups.

    Handles ``.gz`` compressed files transparently.
    """
    file_handle = handle_input_file(path)
    content = file_handle.read()
    file_handle.close()
    return find_na_chain_groups(
        content,
        distance_threshold=distance_threshold,
        atom_name=atom_name,
        model=model,
    )


def _load_cache(path: str) -> Dict[str, Optional[List[List[str]]]]:
    """Load a JSON cache file mapping file paths to chain groups.

    Returns an empty dict if the file does not exist or is malformed.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        if isinstance(data, dict):
            return data
        print(
            f"Warning: cache file {path} is not a JSON object, starting fresh.",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"Warning: failed to read cache file {path}: {exc}. Starting fresh.",
            file=sys.stderr,
        )
    return {}


def _save_cache(path: str, cache: Dict[str, Optional[List[List[str]]]]) -> None:
    """Write the cache dict to a JSON file."""
    with open(path, "wb") as f:
        f.write(orjson.dumps(cache))


def _process_paths(
    paths: List[str],
    threshold: float,
    atom_name: str,
    model: Optional[int],
    cache_path: Optional[str] = None,
    cache: Optional[Dict[str, Optional[List[List[str]]]]] = None,
    save_interval: int = 100,
) -> Dict[str, Optional[List[List[str]]]]:
    """Process a list of file paths and return results.

    Uses a single direct call for one file, or ProcessPoolExecutor with
    a tqdm progress bar for multiple files.

    When ``cache_path`` and ``cache`` are provided, the cache is written
    to disk every ``save_interval`` completed files so progress is not
    lost if the process is interrupted.
    """
    results: Dict[str, Optional[List[List[str]]]] = {}

    if len(paths) == 1:
        path = paths[0]
        try:
            groups = find_na_chain_groups_file(
                path,
                distance_threshold=threshold,
                atom_name=atom_name,
                model=model,
            )
            results[path] = [sorted(g) for g in groups]
        except Exception as exc:
            print(f"Warning: failed to process {path}: {exc}", file=sys.stderr)
            results[path] = None
    else:
        completed = 0
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    find_na_chain_groups_file,
                    path,
                    distance_threshold=threshold,
                    atom_name=atom_name,
                    model=model,
                ): path
                for path in paths
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing",
                unit="file",
            ):
                path = futures[future]
                try:
                    groups = future.result()
                    results[path] = [sorted(g) for g in groups]
                except Exception as exc:
                    print(
                        f"Warning: failed to process {path}: {exc}",
                        file=sys.stderr,
                    )
                    results[path] = None

                completed += 1
                if cache_path and cache is not None and completed % save_interval == 0:
                    cache.update(results)
                    _save_cache(cache_path, dict(sorted(cache.items())))

        results = dict(sorted(results.items()))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly find which nucleic-acid chains are in contact in a "
            "PDB or mmCIF file. Uses a KD-tree on C1' atoms (configurable) "
            "with a distance threshold to group interacting chains."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Path(s) to PDB or mmCIF file(s) (optionally .gz). "
            "If omitted, reads paths from stdin (one per line)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Write results to this JSON file. If the file already exists, "
            "cached entries are reused (files already in the cache are "
            "skipped). New results are merged into the cache and written "
            "back."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help=f"Distance threshold in Angstroms (default: {DEFAULT_DISTANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--atom",
        type=str,
        default=DEFAULT_ATOM_NAME,
        help=f"Representative atom name (default: {DEFAULT_ATOM_NAME!r}).",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=None,
        help="Model number to analyse (default: first model).",
    )
    args = parser.parse_args()

    paths = args.paths
    if not paths:
        paths = [line.strip() for line in sys.stdin if line.strip()]
    if not paths:
        parser.print_help()
        return

    if args.output:
        cache = _load_cache(args.output)
        new_paths = [p for p in paths if p not in cache]
        cached_count = len(paths) - len(new_paths)

        if cached_count > 0:
            print(
                f"Found {cached_count} cached file(s), "
                f"processing {len(new_paths)} new file(s).",
                file=sys.stderr,
            )

        if not new_paths:
            print(f"All {len(paths)} file(s) already cached.", file=sys.stderr)
            return

        new_results = _process_paths(
            new_paths,
            args.threshold,
            args.atom,
            args.model,
            cache_path=args.output,
            cache=cache,
        )
        cache.update(new_results)
        cache = dict(sorted(cache.items()))
        _save_cache(args.output, cache)
    else:
        results = _process_paths(paths, args.threshold, args.atom, args.model)

        if len(results) == 1:
            single = next(iter(results.values()))
            print(orjson.dumps(single).decode("utf-8"))
        else:
            print(orjson.dumps(results).decode("utf-8"))


if __name__ == "__main__":
    main()
