import string
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

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


def calculate_torsion_angle(
    a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, a4: np.ndarray
) -> float:
    """
    Calculate the torsion angle between four points in 3D space.

    Parameters:
    -----------
    a1, a2, a3, a4 : np.ndarray
        3D coordinates of the four atoms

    Returns:
    --------
    float
        Torsion angle in radians
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
    """
    Calculates RMSD using the Quaternion method.

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
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
    """
    Calculates RMSD using SVD decomposition (Kabsch algorithm).

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
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
    """
    Calculates RMSD using the QCP (Quaternion Characteristic Polynomial) method.
    This implementation follows the BioPython QCP algorithm but uses np.linalg.eigh
    instead of Newton-Raphson for simplicity.

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
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
    """
    Convert RMSD to normalized RMSD (nRMSD).

    Parameters:
    -----------
    rmsd : float
        Root Mean Square Deviation value
    num_atoms : int
        Number of atoms used in the RMSD calculation

    Returns:
    --------
    float
        Normalized RMSD value
    """
    return rmsd / np.sqrt(num_atoms)


def nrmsd_quaternions(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculates nRMSD using the Quaternion method.

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
    """
    rmsd = rmsd_quaternions(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_svd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculates nRMSD using SVD decomposition (Kabsch algorithm).

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
    """
    rmsd = rmsd_svd(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_qcp(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculates nRMSD using the QCP (Quaternion Characteristic Polynomial) method.

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure
    """
    rmsd = rmsd_qcp(coords1, coords2)
    return rmsd_to_nrmsd(rmsd, coords1.shape[0])


def nrmsd_validate(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Validates that all nRMSD methods produce the same result.
    Uses quaternions method as the primary result after validation.

    Parameters:
    -----------
    coords1 : np.ndarray
        Nx3 array of coordinates for the first structure
    coords2 : np.ndarray
        Nx3 array of coordinates for the second structure

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
    """
    Calculates nRMSD using the Quaternion method from residue lists.
    residues1 and residues2 are lists of Residue objects.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_quaternions(coords1, coords2)


def nrmsd_svd_residues(residues1: List["Residue"], residues2: List["Residue"]) -> float:
    """
    Calculates nRMSD using SVD decomposition from residue lists.
    residues1 and residues2 are lists of Residue objects.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_svd(coords1, coords2)


def nrmsd_qcp_residues(residues1: List["Residue"], residues2: List["Residue"]) -> float:
    """
    Calculates nRMSD using the QCP method from residue lists.
    residues1 and residues2 are lists of Residue objects.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_qcp(coords1, coords2)


def nrmsd_validate_residues(
    residues1: List["Residue"], residues2: List["Residue"]
) -> float:
    """
    Validates that all nRMSD methods produce the same result from residue lists.
    residues1 and residues2 are lists of Residue objects.
    """
    coords1, coords2 = find_paired_coordinates(residues1, residues2)
    return nrmsd_validate(coords1, coords2)


class Structure:
    """
    A class representing a molecular structure parsed from PDB or mmCIF format.

    This class takes a DataFrame created by parser_v2 functions and provides
    methods to access and manipulate the structure data.
    """

    def __init__(self, atoms: pd.DataFrame):
        """
        Initialize a Structure object with atom data.

        Parameters:
        -----------
        atoms : pd.DataFrame
            DataFrame containing atom data, as created by parse_pdb_atoms or parse_cif_atoms
        """
        self.atoms = atoms
        self.format = atoms.attrs.get("format", "unknown")

    @cached_property
    def residues(self) -> List["Residue"]:
        """
        Group atoms by residue and return a list of Residue objects.

        The grouping logic depends on the format of the input data:
        - For PDB: group by (chainID, resSeq, iCode)
        - For mmCIF: group by (label_asym_id, label_seq_id) if present,
                     otherwise by (auth_asym_id, auth_seq_id, pdbx_PDB_ins_code)

        Returns:
        --------
        List[Residue]
            List of Residue objects, each representing a single residue
        """
        if self.format == "PDB":
            # Group by chain ID, residue sequence number, and insertion code
            groupby_cols = ["chainID", "resSeq", "iCode"]

            # Filter out columns that don't exist in the DataFrame
            groupby_cols = [col for col in groupby_cols if col in self.atoms.columns]

            # Group atoms by residue
            grouped = self.atoms.groupby(groupby_cols, dropna=False, observed=False)

        elif self.format == "mmCIF":
            # Prefer auth_* columns if they exist
            if (
                "auth_asym_id" in self.atoms.columns
                and "auth_seq_id" in self.atoms.columns
            ):
                groupby_cols = ["auth_asym_id", "auth_seq_id"]

                # Add insertion code if it exists
                if "pdbx_PDB_ins_code" in self.atoms.columns:
                    groupby_cols.append("pdbx_PDB_ins_code")
            else:
                # Fall back to label_* columns
                groupby_cols = ["label_asym_id", "label_seq_id"]

                # Add insertion code if it exists
                if "pdbx_PDB_ins_code" in self.atoms.columns:
                    groupby_cols.append("pdbx_PDB_ins_code")

            # Group atoms by residue
            grouped = self.atoms.groupby(
                groupby_cols, dropna=False, observed=False, sort=False
            )

        else:
            # For unknown formats, return an empty list
            return []

        # Convert groups to a list of DataFrames
        residue_dfs = []
        for _, group in grouped:
            # Create a copy of the group DataFrame
            residue_df = group.copy()

            # Preserve the format attribute
            residue_df.attrs["format"] = self.format

            residue_dfs.append(residue_df)

        # Convert groups to a list of Residue objects
        residues = []
        for _, group in grouped:
            # Create a copy of the group DataFrame
            residue_df = group.copy()

            # Preserve the format attribute
            residue_df.attrs["format"] = self.format

            # Create a Residue object
            residues.append(Residue(residue_df))

        return residues

    @cached_property
    def connected_residues(self) -> List[List["Residue"]]:
        """
        Find segments of connected residues in the structure.

        Returns:
        --------
        List[List[Residue]]
            List of segments, where each segment is a list of connected residues
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
        """
        Calculate torsion angles for all connected residues in the structure.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing torsion angle values for each residue
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
                        continue  # Skip alpha for the second residue

                    if angle_name in ["epsilon", "zeta"] and i == len(segment) - 1:
                        continue  # Skip epsilon and zeta for the second-to-last residue

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


class Residue:
    """
    A class representing a single residue in a molecular structure.

    This class encapsulates a DataFrame containing atoms belonging to a single residue
    and provides methods to access residue properties.
    """

    def __init__(self, residue_df: pd.DataFrame):
        """
        Initialize a Residue object with atom data for a single residue.

        Parameters:
        -----------
        residue_df : pd.DataFrame
            DataFrame containing atom data for a single residue
        """
        self.atoms = residue_df
        self.format = residue_df.attrs.get("format", "unknown")

    @property
    def chain_id(self) -> str:
        """Get the chain identifier for this residue."""
        if self.format == "PDB":
            return self.atoms["chainID"].iloc[0]
        elif self.format == "mmCIF":
            if "auth_asym_id" in self.atoms.columns:
                return self.atoms["auth_asym_id"].iloc[0]
            else:
                return self.atoms["label_asym_id"].iloc[0]
        return ""

    @chain_id.setter
    def chain_id(self, value: str) -> None:
        """Set the chain identifier for this residue."""
        if self.format == "PDB":
            self.atoms["chainID"] = value
        elif self.format == "mmCIF":
            if "auth_asym_id" in self.atoms.columns:
                self.atoms["auth_asym_id"] = value
            if "label_asym_id" in self.atoms.columns:
                self.atoms["label_asym_id"] = value

    @property
    def residue_number(self) -> int:
        """Get the residue sequence number."""
        if self.format == "PDB":
            return int(self.atoms["resSeq"].iloc[0])
        elif self.format == "mmCIF":
            if "auth_seq_id" in self.atoms.columns:
                return int(self.atoms["auth_seq_id"].iloc[0])
            else:
                return int(self.atoms["label_seq_id"].iloc[0])
        return 0

    @residue_number.setter
    def residue_number(self, value: int) -> None:
        """Set the residue sequence number."""
        if self.format == "PDB":
            self.atoms["resSeq"] = value
        elif self.format == "mmCIF":
            if "auth_seq_id" in self.atoms.columns:
                self.atoms["auth_seq_id"] = value
            if "label_seq_id" in self.atoms.columns:
                self.atoms["label_seq_id"] = value

    @property
    def insertion_code(self) -> Optional[str]:
        """Get the insertion code, if any."""
        if self.format == "PDB":
            icode = self.atoms["iCode"].iloc[0]
            return icode if pd.notna(icode) else None
        elif self.format == "mmCIF":
            if "pdbx_PDB_ins_code" in self.atoms.columns:
                icode = self.atoms["pdbx_PDB_ins_code"].iloc[0]
                return icode if pd.notna(icode) else None
        return None

    @insertion_code.setter
    def insertion_code(self, value: Optional[str]) -> None:
        """Set the insertion code."""
        if self.format == "PDB":
            self.atoms["iCode"] = value
        elif self.format == "mmCIF":
            if "pdbx_PDB_ins_code" in self.atoms.columns:
                self.atoms["pdbx_PDB_ins_code"] = value

    @cached_property
    def residue_name(self) -> str:
        """Get the residue name (e.g., 'A', 'G', 'C', 'U', etc.)."""
        if self.format == "PDB":
            return self.atoms["resName"].iloc[0]
        elif self.format == "mmCIF":
            if "auth_comp_id" in self.atoms.columns:
                return self.atoms["auth_comp_id"].iloc[0]
            else:
                return self.atoms["label_comp_id"].iloc[0]
        return ""

    @cached_property
    def atoms_list(self) -> List["Atom"]:
        """Get a list of all atoms in this residue."""
        return [Atom(self.atoms.iloc[i], self.format) for i in range(len(self.atoms))]

    @cached_property
    def _atom_dict(self) -> dict[str, "Atom"]:
        """Cache a dictionary of atom names to Atom instances."""
        atom_dict = {}

        for i in range(len(self.atoms)):
            atom_data = self.atoms.iloc[i]
            atom = Atom(atom_data, self.format)

            # Get the atom name based on format
            if self.format == "PDB":
                atom_name = atom_data["name"]
            elif self.format == "mmCIF":
                if "auth_atom_id" in self.atoms.columns:
                    atom_name = atom_data["auth_atom_id"]
                else:
                    atom_name = atom_data["label_atom_id"]
            else:
                continue

            atom_dict[atom_name] = atom

        return atom_dict

    def find_atom(self, atom_name: str) -> Optional["Atom"]:
        """
        Find an atom by name in this residue.

        Parameters:
        -----------
        atom_name : str
            Name of the atom to find

        Returns:
        --------
        Optional[Atom]
            The Atom object, or None if not found
        """
        return self._atom_dict.get(atom_name)

    @cached_property
    def is_nucleotide(self) -> bool:
        """
        Check if this residue is a nucleotide.

        A nucleotide is identified by the presence of specific atoms:
        - Sugar atoms: C1', C2', C3', C4', O4'
        - Base atoms: N1, C2, N3, C4, C5, C6

        Returns:
        --------
        bool
            True if the residue is a nucleotide, False otherwise
        """
        # Early check: if less than 11 atoms, can't be a nucleotide
        if len(self.atoms) < 11:
            return False

        # Required sugar atoms
        sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "O4'"]

        # Required base atoms
        base_atoms = ["N1", "C2", "N3", "C4", "C5", "C6"]

        # Check for all required atoms
        for atom_name in sugar_atoms + base_atoms:
            if self.find_atom(atom_name) is None:
                return False

        return True

    def is_connected(self, next_residue_candidate: "Residue") -> bool:
        """
        Check if this residue is connected to the next residue candidate.

        The connection is determined by the distance between the O3' atom of this residue
        and the P atom of the next residue. If the distance is less than 1.5 times the
        average O-P covalent bond distance, the residues are considered connected.

        Parameters:
        -----------
        next_residue_candidate : Residue
            The residue to check for connection

        Returns:
        --------
        bool
            True if the residues are connected, False otherwise
        """
        o3p = self.find_atom("O3'")
        p = next_residue_candidate.find_atom("P")

        if o3p is not None and p is not None:
            distance = np.linalg.norm(o3p.coordinates - p.coordinates).item()
            return distance < 1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT

        return False

    def __str__(self) -> str:
        """String representation of the residue."""
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
        """Detailed string representation of the residue."""
        return f"Residue({self.__str__()}, {len(self.atoms)} atoms)"


class Atom:
    """
    A class representing a single atom in a molecular structure.

    This class encapsulates a pandas Series containing data for a single atom
    and provides methods to access atom properties.
    """

    def __init__(self, atom_data: pd.Series, format: str):
        """
        Initialize an Atom object with atom data.

        Parameters:
        -----------
        atom_data : pd.Series
            Series containing data for a single atom
        format : str
            Format of the data ('PDB' or 'mmCIF')
        """
        self.data = atom_data
        self.format = format

    @cached_property
    def name(self) -> str:
        """Get the atom name."""
        if self.format == "PDB":
            return self.data["name"]
        elif self.format == "mmCIF":
            if "auth_atom_id" in self.data:
                return self.data["auth_atom_id"]
            else:
                return self.data["label_atom_id"]
        return ""

    @cached_property
    def element(self) -> str:
        """Get the element symbol."""
        if self.format == "PDB":
            return self.data["element"]
        elif self.format == "mmCIF":
            if "type_symbol" in self.data:
                return self.data["type_symbol"]
        return ""

    @cached_property
    def coordinates(self) -> np.ndarray:
        """Get the 3D coordinates of the atom."""
        if self.format == "PDB":
            return np.array([self.data["x"], self.data["y"], self.data["z"]])
        elif self.format == "mmCIF":
            return np.array(
                [self.data["Cartn_x"], self.data["Cartn_y"], self.data["Cartn_z"]]
            )
        return np.array([0.0, 0.0, 0.0])

    @cached_property
    def occupancy(self) -> float:
        """Get the occupancy value."""
        if self.format == "PDB":
            return (
                float(self.data["occupancy"])
                if pd.notna(self.data["occupancy"])
                else 1.0
            )
        elif self.format == "mmCIF":
            if "occupancy" in self.data:
                return (
                    float(self.data["occupancy"])
                    if pd.notna(self.data["occupancy"])
                    else 1.0
                )
        return 1.0

    @cached_property
    def temperature_factor(self) -> float:
        """Get the temperature factor (B-factor)."""
        if self.format == "PDB":
            return (
                float(self.data["tempFactor"])
                if pd.notna(self.data["tempFactor"])
                else 0.0
            )
        elif self.format == "mmCIF":
            if "B_iso_or_equiv" in self.data:
                return (
                    float(self.data["B_iso_or_equiv"])
                    if pd.notna(self.data["B_iso_or_equiv"])
                    else 0.0
                )
        return 0.0

    def __str__(self) -> str:
        """String representation of the atom."""
        return f"{self.name} ({self.element})"

    def __repr__(self) -> str:
        """Detailed string representation of the atom."""
        coords = self.coordinates
        return f"Atom({self.name}, {self.element}, [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}])"
