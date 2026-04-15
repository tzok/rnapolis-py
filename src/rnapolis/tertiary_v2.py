import string
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from rnapolis.common import Molecule, classify_molecule
from rnapolis.parser_v2 import parse_cif_atoms, write_cif

# Constants
AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT = 1.6

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
# DNA backbone atoms (no O2' compared to RNA)
BACKBONE_DEOXYRIBOSE_ATOMS = {
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
}
PURINE_CORE_ATOMS = {"N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"}
PYRIMIDINE_CORE_ATOMS = {"N1", "C2", "N3", "C4", "C5", "C6"}

# RNA nucleotides
ATOMS_A = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS | {"N6"}
ATOMS_G = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS | {"O6"}
ATOMS_C = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"N4", "O2"}
ATOMS_U = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"O4", "O2"}

# DNA nucleotides
ATOMS_DA = BACKBONE_DEOXYRIBOSE_ATOMS | PURINE_CORE_ATOMS | {"N6"}
ATOMS_DG = BACKBONE_DEOXYRIBOSE_ATOMS | PURINE_CORE_ATOMS | {"O6"}
ATOMS_DC = BACKBONE_DEOXYRIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"N4", "O2"}
ATOMS_DT = BACKBONE_DEOXYRIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS | {"O4", "O2", "C7"}

PURINES = {"A", "G", "DA", "DG"}
PYRIMIDINES = {"C", "U", "DC", "DT"}
RESIDUE_ATOMS_MAP = {
    "A": ATOMS_A,
    "G": ATOMS_G,
    "C": ATOMS_C,
    "U": ATOMS_U,
    "DA": ATOMS_DA,
    "DG": ATOMS_DG,
    "DC": ATOMS_DC,
    "DT": ATOMS_DT,
}
REQUIRED_NUCLEOTIDE_ATOMS = frozenset(
    {
        "C1'",
        "C2'",
        "C3'",
        "C4'",
        "O4'",
        "N1",
        "C2",
        "N3",
        "C4",
        "C5",
        "C6",
    }
)
AMINO_ACID_NAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
    "MSE",
    "UNK",
}

# ---------------------------------------------------------------------------
# Column maps: semantic name -> format-specific column name(s)
# ---------------------------------------------------------------------------
# For mmCIF, a tuple means (preferred, fallback).  For PDB, always a plain str.

_ATOM_COLUMN_MAP: Dict[str, Dict[str, Union[str, Tuple[str, str]]]] = {
    "record_type": {"PDB": "record_type", "mmCIF": "group_PDB"},
    "serial": {"PDB": "serial", "mmCIF": "id"},
    "atom_name": {"PDB": "name", "mmCIF": ("auth_atom_id", "label_atom_id")},
    "alt_loc": {"PDB": "altLoc", "mmCIF": "label_alt_id"},
    "residue_name": {"PDB": "resName", "mmCIF": ("auth_comp_id", "label_comp_id")},
    "chain_id": {"PDB": "chainID", "mmCIF": ("auth_asym_id", "label_asym_id")},
    "residue_number": {"PDB": "resSeq", "mmCIF": ("auth_seq_id", "label_seq_id")},
    "insertion_code": {"PDB": "iCode", "mmCIF": "pdbx_PDB_ins_code"},
    "x": {"PDB": "x", "mmCIF": "Cartn_x"},
    "y": {"PDB": "y", "mmCIF": "Cartn_y"},
    "z": {"PDB": "z", "mmCIF": "Cartn_z"},
    "occupancy": {"PDB": "occupancy", "mmCIF": "occupancy"},
    "b_factor": {"PDB": "tempFactor", "mmCIF": "B_iso_or_equiv"},
    "element": {"PDB": "element", "mmCIF": "type_symbol"},
    "charge": {"PDB": "charge", "mmCIF": "pdbx_formal_charge"},
    "model": {"PDB": "model", "mmCIF": "pdbx_PDB_model_num"},
}

_MODRES_COLUMN_MAP: Dict[str, Dict[str, Union[str, Tuple[str, str]]]] = {
    "residue_name": {"PDB": "resName", "mmCIF": "auth_comp_id"},
    "chain_id": {"PDB": "chainID", "mmCIF": "auth_asym_id"},
    "residue_number": {"PDB": "seqNum", "mmCIF": "auth_seq_id"},
    "insertion_code": {"PDB": "iCode", "mmCIF": "pdbx_PDB_ins_code"},
    "standard_name": {"PDB": "stdRes", "mmCIF": "parent_comp_id"},
    "comment": {"PDB": "comment", "mmCIF": "details"},
}


class _ColumnAccessorMixin:
    """Resolves semantic column names to actual DataFrame columns.

    At init, resolves mmCIF fallbacks (auth_* -> label_*) once and caches the
    result.  After that, :meth:`_col` is a simple dict lookup.
    """

    format: str

    def _init_columns(
        self,
        available: Union[pd.Index, "frozenset[str]", "set[str]"],
        fmt: str,
        column_map: Dict[str, Dict[str, Union[str, Tuple[str, str]]]],
    ) -> None:
        """Resolve and cache column names for *fmt*.

        Parameters
        ----------
        available:
            Column (or index) names present in the underlying data.
            For a DataFrame pass ``df.columns``; for a Series pass
            ``series.index``.
        fmt:
            Data format, e.g. ``"PDB"`` or ``"mmCIF"``.
        column_map:
            Semantic-name -> per-format column specification.
        """
        resolved: Dict[str, str] = {}
        for semantic_name, format_entry in column_map.items():
            spec = format_entry.get(fmt)
            if spec is None:
                continue
            if isinstance(spec, tuple):
                preferred, fallback = spec
                if preferred in available:
                    resolved[semantic_name] = preferred
                elif fallback in available:
                    resolved[semantic_name] = fallback
            else:
                if spec in available:
                    resolved[semantic_name] = spec
        self._resolved_columns = resolved

    def _col(self, key: str) -> str:
        """Return the actual DataFrame column name for a semantic *key*."""
        try:
            return self._resolved_columns[key]
        except KeyError:
            raise KeyError(
                f"Column '{key}' could not be resolved for format '{self.format}'"
            ) from None

    def _write_cols(
        self,
        key: str,
        available: Union[pd.Index, "frozenset[str]", "set[str]"],
        column_map: Dict[str, Dict[str, Union[str, Tuple[str, str]]]],
    ) -> List[str]:
        """Return all DataFrame columns that should be written for *key*.

        For PDB (or any single-column spec), returns the one resolved column.
        For mmCIF tuple specs (preferred, fallback), returns every column from
        the spec that actually exists in *available* — so both ``auth_*`` and
        ``label_*`` get updated when they are present.
        """
        spec = column_map.get(key, {}).get(self.format)
        if spec is None:
            return []
        if isinstance(spec, tuple):
            return [c for c in spec if c in available]
        return [spec] if spec in available else []


def calculate_torsion_angle(
    a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, a4: np.ndarray
) -> float:
    """Calculate the torsion (dihedral) angle between four 3D points.

    Args:
        a1: Coordinates of the first atom (x, y, z).
        a2: Coordinates of the second atom (x, y, z).
        a3: Coordinates of the third atom (x, y, z).
        a4: Coordinates of the fourth atom (x, y, z).

    Returns:
        Torsion angle in radians.
    """
    # Calculate vectors between points
    v1 = a2 - a1
    v2 = a3 - a2
    v3 = a4 - a3

    # Calculate normal vectors
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Normalize normal vectors
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    # Check for collinearity
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return float("nan")

    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Calculate the angle using dot product
    m1 = np.cross(n1, v2 / np.linalg.norm(v2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    # Return angle in radians
    angle = np.arctan2(y, x)

    return angle


def find_paired_coordinates(
    residues1: List["Residue"], residues2: List["Residue"]
) -> Tuple[np.ndarray, np.ndarray]:
    """Find matching atom coordinates between two residue lists.

    For each pair of residues, the function selects a consistent set of atoms
    (RNA/DNA backbone + base core) and returns coordinates of matching atoms
    from both structures.

    Args:
        residues1: List of residues from the first structure.
        residues2: List of residues from the second structure.

    Returns:
        A tuple (coords_1, coords_2) with two arrays of shape (N, 3) containing coordinates of corresponding atoms.
    """
    all_paired_dfs = []

    for residue1, residue2 in zip(residues1, residues2):
        res_name1 = residue1.residue_name
        res_name2 = residue2.residue_name

        atoms_to_match = None

        if res_name1 == res_name2:
            atoms_to_match = RESIDUE_ATOMS_MAP.get(res_name1)
        elif res_name1 in PURINES and res_name2 in PURINES:
            # For mixed RNA/DNA purines, use common backbone + purine core
            if any(name.startswith("D") for name in [res_name1, res_name2]):
                # At least one is DNA, use deoxyribose backbone
                atoms_to_match = BACKBONE_DEOXYRIBOSE_ATOMS | PURINE_CORE_ATOMS
            else:
                # Both RNA, use ribose backbone
                atoms_to_match = BACKBONE_RIBOSE_ATOMS | PURINE_CORE_ATOMS
        elif res_name1 in PYRIMIDINES and res_name2 in PYRIMIDINES:
            # For mixed RNA/DNA pyrimidines, use common backbone + pyrimidine core
            if any(name.startswith("D") for name in [res_name1, res_name2]):
                # At least one is DNA, use deoxyribose backbone
                atoms_to_match = BACKBONE_DEOXYRIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS
            else:
                # Both RNA, use ribose backbone
                atoms_to_match = BACKBONE_RIBOSE_ATOMS | PYRIMIDINE_CORE_ATOMS
        else:
            # Different types, use minimal common backbone
            if any(name.startswith("D") for name in [res_name1, res_name2]):
                atoms_to_match = BACKBONE_DEOXYRIBOSE_ATOMS
            else:
                atoms_to_match = BACKBONE_RIBOSE_ATOMS

        # Ensure atoms_to_match is not None
        if atoms_to_match is None:
            # Fallback to minimal backbone atoms
            atoms_to_match = BACKBONE_DEOXYRIBOSE_ATOMS

        if residue1.format == "mmCIF":
            df1 = residue1.atoms
        else:
            df1 = parse_cif_atoms(write_cif(residue1.atoms))

        if residue2.format == "mmCIF":
            df2 = residue2.atoms
        else:
            df2 = parse_cif_atoms(write_cif(residue2.atoms))

        if "auth_atom_id" in df1.columns and "auth_atom_id" in df2.columns:
            atom_column = "auth_atom_id"
        elif "label_atom_id" in df1.columns and "label_atom_id" in df2.columns:
            atom_column = "label_atom_id"
        else:
            raise ValueError(
                "No suitable atom identifier column found in the provided residues."
            )

        df1_filtered = df1[df1[atom_column].isin(atoms_to_match)]
        df2_filtered = df2[df2[atom_column].isin(atoms_to_match)]

        paired_df = pd.merge(
            df1_filtered[[atom_column, "Cartn_x", "Cartn_y", "Cartn_z"]],
            df2_filtered[[atom_column, "Cartn_x", "Cartn_y", "Cartn_z"]],
            on=atom_column,
            suffixes=("_1", "_2"),
        )

        if not paired_df.empty:
            all_paired_dfs.append(paired_df)

    final_df = pd.concat(all_paired_dfs, ignore_index=True)
    coords_1 = final_df[["Cartn_x_1", "Cartn_y_1", "Cartn_z_1"]].to_numpy()
    coords_2 = final_df[["Cartn_x_2", "Cartn_y_2", "Cartn_z_2"]].to_numpy()
    return coords_1, coords_2


def rmsd_quaternions(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD using the quaternion-based method.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Root mean square deviation between coords1 and coords2.
    """
    P, Q = coords1, coords2

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
    return np.sqrt(max(0.0, rmsd_sq))


def rmsd_svd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD using SVD (Kabsch algorithm).

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Root mean square deviation between coords1 and coords2.
    """
    P, Q = coords1, coords2

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

    return np.sqrt(rmsd_sq)


def rmsd_qcp(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD using the QCP (Quaternion Characteristic Polynomial) method.

    This implementation follows the BioPython QCP algorithm but uses
    ``np.linalg.eigh`` to find the largest eigenvalue.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Root mean square deviation between coords1 and coords2.
    """

    # Center coordinates at origin
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2

    # Calculate G1, G2, and cross-covariance matrix A (following BioPython)
    G1 = np.trace(np.dot(coords2_centered, coords2_centered.T))
    G2 = np.trace(np.dot(coords1_centered, coords1_centered.T))
    A = np.dot(coords2_centered.T, coords1_centered)  # Cross-covariance matrix
    E0 = (G1 + G2) * 0.5

    # Extract elements from A matrix
    Sxx, Sxy, Sxz = A[0, 0], A[0, 1], A[0, 2]
    Syx, Syy, Syz = A[1, 0], A[1, 1], A[1, 2]
    Szx, Szy, Szz = A[2, 0], A[2, 1], A[2, 2]

    # Build the K matrix (quaternion matrix) as in BioPython
    K = np.zeros((4, 4))
    K[0, 0] = Sxx + Syy + Szz
    K[0, 1] = K[1, 0] = Syz - Szy
    K[0, 2] = K[2, 0] = Szx - Sxz
    K[0, 3] = K[3, 0] = Sxy - Syx
    K[1, 1] = Sxx - Syy - Szz
    K[1, 2] = K[2, 1] = Sxy + Syx
    K[1, 3] = K[3, 1] = Szx + Sxz
    K[2, 2] = -Sxx + Syy - Szz
    K[2, 3] = K[3, 2] = Syz + Szy
    K[3, 3] = -Sxx - Syy + Szz

    # Find the largest eigenvalue using numpy
    eigenvalues, _ = np.linalg.eigh(K)
    max_eigenvalue = np.max(eigenvalues)

    # Calculate RMSD following BioPython formula
    natoms = coords1.shape[0]
    rmsd_sq = (2.0 * abs(E0 - max_eigenvalue)) / natoms
    rmsd = np.sqrt(rmsd_sq)

    return rmsd


def rmsd_to_nrmsd(rmsd: float, num_atoms: int) -> float:
    """Convert RMSD to normalized RMSD (nRMSD).

    Args:
        rmsd: RMSD value.
        num_atoms: Number of atoms used to compute the RMSD.

    Returns:
        Normalized RMSD.
    """
    return rmsd / np.sqrt(num_atoms)


def nrmsd_quaternions(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate nRMSD using the quaternion method.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Normalized RMSD value.
    """
    rmsd = rmsd_quaternions(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_svd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate nRMSD using the SVD (Kabsch) method.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Normalized RMSD value.
    """
    rmsd = rmsd_svd(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_qcp(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate nRMSD using the QCP method.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        Normalized RMSD value.
    """
    rmsd = rmsd_qcp(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_validate(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Validate that all nRMSD implementations agree.

    Calculates nRMSD using three methods (quaternions, SVD, QCP) and checks
    that they are numerically consistent. Returns the quaternion result.

    Args:
        coords1: Array of shape (N, 3) with coordinates of the first structure.
        coords2: Array of shape (N, 3) with coordinates of the second structure.

    Returns:
        nRMSD value from the quaternion method.

    Raises:
        ValueError: If any pair of methods differs by more than the tolerance.
    """
    # Calculate using all methods
    result_quaternions = nrmsd_quaternions(coords1, coords2)
    result_svd = nrmsd_svd(coords1, coords2)
    result_qcp = nrmsd_qcp(coords1, coords2)

    # Check if results are approximately equal (within 1e-6 tolerance)
    tolerance = 1e-6

    # Check quaternions vs SVD
    if abs(result_quaternions - result_svd) > tolerance:
        raise ValueError(
            f"nRMSD methods disagree: quaternions={result_quaternions:.8f}, "
            f"svd={result_svd:.8f}, difference={abs(result_quaternions - result_svd):.8f}"
        )

    # Check quaternions vs QCP
    if abs(result_quaternions - result_qcp) > tolerance:
        raise ValueError(
            f"nRMSD methods disagree: quaternions={result_quaternions:.8f}, "
            f"qcp={result_qcp:.8f}, difference={abs(result_quaternions - result_qcp):.8f}"
        )

    # Check SVD vs QCP
    if abs(result_svd - result_qcp) > tolerance:
        raise ValueError(
            f"nRMSD methods disagree: svd={result_svd:.8f}, "
            f"qcp={result_qcp:.8f}, difference={abs(result_svd - result_qcp):.8f}"
        )

    # Return quaternions result as the validated value
    return result_quaternions


def nrmsd_quaternions_residues(
    residues1: List["Residue"], residues2: List["Residue"]
) -> float:
    """Calculate nRMSD (quaternion) directly from residue lists.

    Args:
        residues1: Residues from the first structure.
        residues2: Residues from the second structure.

    Returns:
        Normalized RMSD value.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_quaternions(coords1, coords2)


def nrmsd_svd_residues(residues1: List["Residue"], residues2: List["Residue"]) -> float:
    """Calculate nRMSD (SVD) directly from residue lists.

    Args:
        residues1: Residues from the first structure.
        residues2: Residues from the second structure.

    Returns:
        Normalized RMSD value.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_svd(coords1, coords2)


def nrmsd_qcp_residues(residues1: List["Residue"], residues2: List["Residue"]) -> float:
    """Calculate nRMSD (QCP) directly from residue lists.

    Args:
        residues1: Residues from the first structure.
        residues2: Residues from the second structure.

    Returns:
        Normalized RMSD value.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_qcp(coords1, coords2)


def nrmsd_validate_residues(
    residues1: List["Residue"], residues2: List["Residue"]
) -> float:
    """Validate that all nRMSD methods agree for residue lists.

    Args:
        residues1: Residues from the first structure.
        residues2: Residues from the second structure.

    Returns:
        nRMSD value from the quaternion method.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_validate(coords1, coords2)


class ModifiedResidues(_ColumnAccessorMixin):
    """Wrapper around a MODRES DataFrame with format-agnostic lookup.

    Uses :class:`_ColumnAccessorMixin` with :data:`_MODRES_COLUMN_MAP` so that
    callers never need to know whether the underlying data came from a PDB
    MODRES section or an mmCIF ``_pdbx_struct_mod_residue`` category.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize from a MODRES DataFrame.

        Args:
            df: DataFrame produced by ``parse_pdb_modres`` or
                ``parse_cif_modres``.  Must carry ``attrs["format"]``.
        """
        self.df = df
        self.format = df.attrs.get("format", "unknown")
        self._init_columns(df.columns, self.format, _MODRES_COLUMN_MAP)

    @property
    def empty(self) -> bool:
        """Return True if there are no MODRES records."""
        return self.df.empty

    def lookup(
        self,
        chain_id: str,
        residue_number: int,
        insertion_code: str,
        residue_name: str,
    ) -> Optional[str]:
        """Look up the standard parent residue name for a modified residue.

        Args:
            chain_id: Chain identifier of the residue.
            residue_number: Sequence number of the residue.
            insertion_code: Insertion code (use ``""`` if none).
            residue_name: The (possibly modified) residue name.

        Returns:
            Standard residue name if a MODRES mapping exists and is non-empty,
            otherwise ``None``.
        """
        if self.df.empty:
            return None

        chain_col = self._col("chain_id")
        resnum_col = self._col("residue_number")
        icode_col = self._col("insertion_code")
        resname_col = self._col("residue_name")
        stdname_col = self._col("standard_name")

        mask = (
            (self.df[chain_col].astype(str) == chain_id)
            & (self.df[resnum_col] == residue_number)
            & (self.df[icode_col] == insertion_code)
            & (self.df[resname_col].astype(str) == residue_name)
        )

        matching = self.df[mask]
        if not matching.empty:
            std_res = matching.iloc[0][stdname_col]
            if pd.notna(std_res) and std_res:
                return str(std_res)

        return None


class Structure(_ColumnAccessorMixin):
    """Molecular structure parsed from PDB or mmCIF.

    Wraps a DataFrame of atoms (from parser_v2) and exposes convenient
    accessors for residues, connected segments and backbone torsion angles.
    """

    def __init__(
        self,
        atoms: pd.DataFrame,
        modres: Optional["ModifiedResidues"] = None,
    ):
        """Initialize a Structure with atom coordinates and metadata.

        Args:
            atoms: DataFrame created by ``parse_pdb_atoms`` or ``parse_cif_atoms``.
            modres: Optional :class:`ModifiedResidues` wrapper, created from
                ``parse_pdb_modres`` or ``parse_cif_modres`` output.
        """
        self.atoms = atoms
        self.format = atoms.attrs.get("format", "unknown")
        self.modres = modres
        self._init_columns(atoms.columns, self.format, _ATOM_COLUMN_MAP)

    @cached_property
    def residues(self) -> List["Residue"]:
        """Group atoms into residues and return them as Residue objects.

        Grouping rules:

        - PDB: by (chainID, resSeq, iCode)
        - mmCIF: by (auth_asym_id, auth_seq_id, pdbx_PDB_ins_code) if present,
          otherwise by (label_asym_id, label_seq_id, pdbx_PDB_ins_code).

        Returns:
            List of Residue objects.
        """
        if self.format == "unknown":
            return []

        groupby_cols = [self._col("chain_id"), self._col("residue_number")]

        # Insertion code is optional (not all formats / files have it)
        if "insertion_code" in self._resolved_columns:
            groupby_cols.append(self._col("insertion_code"))

        # mmCIF needs sort=False to preserve file order
        sort = self.format == "PDB"
        grouped = self.atoms.groupby(
            groupby_cols, dropna=False, observed=True, sort=sort
        )

        # Convert groups to a list of Residue objects
        residues = []
        for _, group in grouped:
            residue_df = group.copy()
            residue_df.attrs["format"] = self.format
            residues.append(Residue(residue_df, self.modres))

        return residues

    @cached_property
    def connected_residues(self) -> List[List["Residue"]]:
        """Find segments of covalently connected residues.

        Residues are grouped by chain and sorted by residue number; within each
        chain, segments of sequentially connected residues (via O3'–P) are returned.

        Returns:
            List of segments; each segment is a list of Residue objects.
        """
        # Group residues by chain
        residues_by_chain = {}
        for residue in self.residues:
            chain_id = residue.chain_id
            if chain_id not in residues_by_chain:
                residues_by_chain[chain_id] = []
            residues_by_chain[chain_id].append(residue)

        # Sort residues in each chain by residue number
        for chain_id in residues_by_chain:
            residues_by_chain[chain_id].sort(
                key=lambda r: (r.residue_number, r.insertion_code or "")
            )

        # Find connected segments in each chain
        segments = []
        for chain_id, chain_residues in residues_by_chain.items():
            current_segment = []

            for residue in chain_residues:
                if not current_segment:
                    # Start a new segment
                    current_segment.append(residue)
                else:
                    # Check if this residue is connected to the previous one
                    prev_residue = current_segment[-1]
                    if prev_residue.is_connected(residue):
                        current_segment.append(residue)
                    else:
                        # End the current segment and start a new one
                        if (
                            len(current_segment) > 1
                        ):  # Only add segments with at least 2 residues
                            segments.append(current_segment)
                        current_segment = [residue]

            # Add the last segment if it has at least 2 residues
            if len(current_segment) > 1:
                segments.append(current_segment)

        return segments

    @cached_property
    def torsion_angles(self) -> pd.DataFrame:
        """Compute backbone and chi torsion angles for all connected residues.

        For each residue in each connected segment, the following torsions
        are calculated when possible:

        - alpha, beta, gamma, delta, epsilon, zeta
        - chi (purine and pyrimidine definitions)

        Returns:
            DataFrame with one row per residue and columns: chain_id, residue_number, insertion_code, residue_name, alpha, beta, gamma, delta, epsilon, zeta, chi.
        """
        # Find connected segments
        segments = self.connected_residues

        # Prepare data for the DataFrame
        data = []

        # Define the torsion angles to calculate
        torsion_definitions = {
            "alpha": [("O3'", -1), ("P", 0), ("O5'", 0), ("C5'", 0)],
            "beta": [("P", 0), ("O5'", 0), ("C5'", 0), ("C4'", 0)],
            "gamma": [("O5'", 0), ("C5'", 0), ("C4'", 0), ("C3'", 0)],
            "delta": [("C5'", 0), ("C4'", 0), ("C3'", 0), ("O3'", 0)],
            "epsilon": [("C4'", 0), ("C3'", 0), ("O3'", 0), ("P", 1)],
            "zeta": [("C3'", 0), ("O3'", 0), ("P", 1), ("O5'", 1)],
            "chi": None,  # Will be handled separately due to purine/pyrimidine difference
        }

        # Process each segment
        for segment in segments:
            for i, residue in enumerate(segment):
                # Prepare row data
                row = {
                    "chain_id": residue.chain_id,
                    "residue_number": residue.residue_number,
                    "insertion_code": residue.insertion_code,
                    "residue_name": residue.residue_name,
                }

                # Calculate standard torsion angles
                for angle_name, atoms_def in torsion_definitions.items():
                    if angle_name == "chi":
                        continue  # Skip chi for now

                    if angle_name == "alpha" and i == 0:
                        continue  # Skip alpha for the first residue

                    if angle_name in ["epsilon", "zeta"] and i == len(segment) - 1:
                        continue  # Skip epsilon and zeta for the last residue

                    # Get the atoms for this angle
                    atoms = []
                    valid = True

                    for atom_name, offset in atoms_def:
                        res_idx = i + offset
                        if 0 <= res_idx < len(segment):
                            atom = segment[res_idx].find_atom(atom_name)
                            if atom is not None:
                                atoms.append(atom.coordinates)
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break

                    # Calculate the angle if all atoms were found
                    if valid and len(atoms) == 4:
                        angle = calculate_torsion_angle(
                            atoms[0], atoms[1], atoms[2], atoms[3]
                        )
                        row[angle_name] = angle
                    else:
                        row[angle_name] = None

                # Calculate chi angle based on residue type
                # Pyrimidines: O4'-C1'-N1-C2
                # Purines: O4'-C1'-N9-C4
                purine_bases = ["A", "G", "DA", "DG"]
                pyrimidine_bases = ["C", "U", "T", "DC", "DT"]

                o4_prime = residue.find_atom("O4'")
                c1_prime = residue.find_atom("C1'")

                if o4_prime is not None and c1_prime is not None:
                    if residue.residue_name in purine_bases:
                        n9 = residue.find_atom("N9")
                        c4 = residue.find_atom("C4")
                        if n9 is not None and c4 is not None:
                            chi = calculate_torsion_angle(
                                o4_prime.coordinates,
                                c1_prime.coordinates,
                                n9.coordinates,
                                c4.coordinates,
                            )
                            row["chi"] = chi
                    elif residue.residue_name in pyrimidine_bases:
                        n1 = residue.find_atom("N1")
                        c2 = residue.find_atom("C2")
                        if n1 is not None and c2 is not None:
                            chi = calculate_torsion_angle(
                                o4_prime.coordinates,
                                c1_prime.coordinates,
                                n1.coordinates,
                                c2.coordinates,
                            )
                            row["chi"] = chi

                data.append(row)

        # Create DataFrame
        if not data:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(
                columns=[
                    "chain_id",
                    "residue_number",
                    "insertion_code",
                    "residue_name",
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "chi",
                ]
            )

        df = pd.DataFrame(data)

        # Ensure all angle columns exist
        for angle in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]:
            if angle not in df.columns:
                df[angle] = None

        # Reorder columns to ensure consistent order
        ordered_columns = [
            "chain_id",
            "residue_number",
            "insertion_code",
            "residue_name",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "chi",
        ]
        df = df[ordered_columns]

        return df


class Residue(_ColumnAccessorMixin):
    """Single residue in a molecular structure.

    Wraps a DataFrame with atoms belonging to one residue and exposes
    basic properties like chain ID, residue number, name and connectivity.
    """

    def __init__(
        self,
        residue_df: pd.DataFrame,
        modres: Optional["ModifiedResidues"] = None,
    ):
        """Initialize a Residue from a DataFrame with atom records.

        Args:
            residue_df: DataFrame containing atom data for a single residue.
            modres: Optional :class:`ModifiedResidues` wrapper with modified
                residue mappings.
        """
        self.atoms = residue_df
        self.format = residue_df.attrs.get("format", "unknown")
        self.modres = modres
        self._init_columns(residue_df.columns, self.format, _ATOM_COLUMN_MAP)

    @property
    def chain_id(self) -> str:
        """Return the chain identifier for this residue."""
        return self.atoms[self._col("chain_id")].iloc[0]

    @chain_id.setter
    def chain_id(self, value: str) -> None:
        """Set the chain identifier for this residue."""
        for col in self._write_cols("chain_id", self.atoms.columns, _ATOM_COLUMN_MAP):
            self.atoms[col] = value

    @property
    def residue_number(self) -> int:
        """Return the residue sequence number."""
        return int(self.atoms[self._col("residue_number")].iloc[0])

    @residue_number.setter
    def residue_number(self, value: int) -> None:
        """Set the residue sequence number."""
        for col in self._write_cols(
            "residue_number", self.atoms.columns, _ATOM_COLUMN_MAP
        ):
            self.atoms[col] = value

    @property
    def insertion_code(self) -> Optional[str]:
        """Return the insertion code, if present."""
        if "insertion_code" not in self._resolved_columns:
            return None
        icode = self.atoms[self._col("insertion_code")].iloc[0]
        return icode if pd.notna(icode) else None

    @insertion_code.setter
    def insertion_code(self, value: Optional[str]) -> None:
        """Set the insertion code."""
        for col in self._write_cols(
            "insertion_code", self.atoms.columns, _ATOM_COLUMN_MAP
        ):
            self.atoms[col] = value

    @cached_property
    def residue_name(self) -> str:
        """Return the residue name (e.g. A, G, C, U, DA...)."""
        return self.atoms[self._col("residue_name")].iloc[0]

    @cached_property
    def standard_residue_name(self) -> str:
        """Return the standard residue name, looking up MODRES if available.

        If this residue is found in the MODRES mapping, returns the standard
        name. Otherwise returns the original residue_name.
        """
        if self.modres is None or self.modres.empty:
            return self.residue_name

        result = self.modres.lookup(
            self.chain_id,
            self.residue_number,
            self.insertion_code or "",
            self.residue_name,
        )
        return result if result is not None else self.residue_name

    @cached_property
    def molecule_type(self) -> Molecule:
        """Classify residue as RNA, DNA or Other.

        Delegates to :func:`~rnapolis.common.classify_molecule`, passing the
        standard residue name and the set of atom names present in this residue.
        """
        return classify_molecule(
            self.standard_residue_name, frozenset(self._atom_names)
        )

    @cached_property
    def one_letter_name(self) -> str:
        """
        Get the one-letter name for the residue.

        If the residue is a nucleotide, it attempts to match the atom set
        against known RNA/DNA bases (A, C, G, U, DA, DC, DG, DT).
        Returns lowercase for DNA bases (a, c, g, t).
        Returns the first letter of the residue name otherwise.
        """
        if not self.is_nucleotide:
            return self.residue_name[0] if self.residue_name else "?"

        # Get the set of atom names present in this residue
        present_atom_names = set(self._atom_dict.keys())

        best_match = None
        max_overlap = -1

        # Iterate over all known nucleotide types
        for res_type, required_atoms in RESIDUE_ATOMS_MAP.items():
            # Calculate overlap: number of required atoms that are present
            overlap = len(present_atom_names.intersection(required_atoms))

            # We prioritize the match that has the highest number of required atoms present.
            # If overlap is equal, we keep the first one found (which is fine for this purpose).
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = res_type

        if best_match is None:
            return self.residue_name[0] if self.residue_name else "?"

        # Map the best match to the one-letter code
        if best_match in {"A", "G", "C", "U"}:
            return best_match
        elif best_match.startswith("D"):
            # DA, DG, DC, DT -> a, g, c, t
            return best_match[1].lower()
        else:
            # Fallback for unknown types, should not happen if RESIDUE_ATOMS_MAP is complete
            return best_match[0]

    @cached_property
    def atoms_list(self) -> List["Atom"]:
        """Return all atoms in this residue as Atom objects."""
        return [Atom(self.atoms.iloc[i], self.format) for i in range(len(self.atoms))]

    @cached_property
    def _atom_dict(self) -> dict[str, "Atom"]:
        """Map atom names to Atom objects for fast lookup."""
        atom_name_col = self._col("atom_name")
        atom_dict = {}

        for i in range(len(self.atoms)):
            atom_data = self.atoms.iloc[i]
            atom = Atom(atom_data, self.format)
            atom_dict[atom_data[atom_name_col]] = atom

        return atom_dict

    @cached_property
    def _atom_names(self) -> set[str]:
        """Cache the set of atom names in this residue."""
        atom_names = set()
        series = self.atoms[self._col("atom_name")]

        for atom_name in series:
            if pd.isna(atom_name):
                continue
            atom_name = str(atom_name)
            if not atom_name or atom_name in {"?", "."}:
                continue
            atom_names.add(atom_name)

        return atom_names

    def find_atom(self, atom_name: str) -> Optional["Atom"]:
        """Find an atom by name in this residue.

        Args:
            atom_name: Name of the atom (e.g. "C1'", "N1").

        Returns:
            Atom object if present, otherwise None.
        """
        return self._atom_dict.get(atom_name)

    @cached_property
    def is_nucleotide(self) -> bool:
        """Check whether this residue looks like a nucleotide.

        A nucleotide is identified by the presence of:

        - sugar atoms: C1', C2', C3', C4', O4'
        - base atoms: N1, C2, N3, C4, C5, C6

        Returns:
            True if all required atoms are present, False otherwise.
        """
        atom_names = self._atom_names
        if len(atom_names) < len(REQUIRED_NUCLEOTIDE_ATOMS):
            return False

        return REQUIRED_NUCLEOTIDE_ATOMS.issubset(atom_names)

    @cached_property
    def is_amino_acid(self) -> bool:
        """Check if this residue is a standard amino acid."""
        residue_name = self.residue_name.strip().upper()
        return residue_name in AMINO_ACID_NAMES

    def is_connected(self, next_residue_candidate: "Residue") -> bool:
        """Check whether this residue is covalently connected to the next.

        The connection is defined by the distance between:

        - O3' atom of this residue and
        - P atom of the next residue.

        If the distance is less than 1.5 × average O–P covalent bond
        distance, residues are considered connected.

        Args:
            next_residue_candidate: Residue to check against.

        Returns:
            True if residues are connected, False otherwise.
        """
        o3p = self.find_atom("O3'")
        p = next_residue_candidate.find_atom("P")

        if o3p is not None and p is not None:
            distance = np.linalg.norm(o3p.coordinates - p.coordinates).item()
            return distance < 1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT

        return False

    def __str__(self) -> str:
        """Return a compact human-readable identifier for the residue."""
        # Start with chain ID and residue name
        chain = self.chain_id
        if chain.isspace() or not chain:
            builder = f"{self.residue_name}"
        else:
            builder = f"{chain}.{self.residue_name}"

        # Add a separator if the residue name ends with a digit
        if len(self.residue_name) > 0 and self.residue_name[-1] in string.digits:
            builder += "/"

        # Add residue number
        builder += f"{self.residue_number}"

        # Add insertion code if present
        icode = self.insertion_code
        if icode is not None:
            builder += f"^{icode}"

        return builder

    def __repr__(self) -> str:
        """Return a detailed string representation with atom count."""
        return f"Residue({self.__str__()}, {len(self.atoms)} atoms)"


class Atom(_ColumnAccessorMixin):
    """Single atom in a molecular structure.

    Wraps a pandas Series with atom data and exposes basic properties such as
    name, element, coordinates, occupancy and B-factor.
    """

    def __init__(self, atom_data: pd.Series, format: str):
        """Initialize an Atom.

        Args:
            atom_data: Series containing data for a single atom.
            format: Data format, e.g. "PDB" or "mmCIF".
        """
        self.data = atom_data
        self.format = format
        self._init_columns(atom_data.index, self.format, _ATOM_COLUMN_MAP)

    @cached_property
    def name(self) -> str:
        """Return the atom name (e.g. C1', N1, O3')."""
        return self.data[self._col("atom_name")]

    @cached_property
    def element(self) -> str:
        """Return the element symbol (e.g. C, N, O, P)."""
        if "element" not in self._resolved_columns:
            return ""
        return self.data[self._col("element")]

    @cached_property
    def coordinates(self) -> np.ndarray:
        """Return atom coordinates as a NumPy array [x, y, z]."""
        return np.array(
            [
                self.data[self._col("x")],
                self.data[self._col("y")],
                self.data[self._col("z")],
            ]
        )

    @cached_property
    def occupancy(self) -> float:
        """Return atom occupancy (defaults to 1.0 if missing)."""
        if "occupancy" not in self._resolved_columns:
            return 1.0
        val = self.data[self._col("occupancy")]
        return float(val) if pd.notna(val) else 1.0

    @cached_property
    def temperature_factor(self) -> float:
        """Return the B-factor (temperature factor) for this atom."""
        if "b_factor" not in self._resolved_columns:
            return 0.0
        val = self.data[self._col("b_factor")]
        return float(val) if pd.notna(val) else 0.0

    def __str__(self) -> str:
        """Return a compact representation ``'NAME (ELEMENT)'``."""
        return f"{self.name} ({self.element})"

    def __repr__(self) -> str:
        """Return a detailed representation including coordinates."""
        coords = self.coordinates
        return f"Atom({self.name}, {self.element}, [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}])"
