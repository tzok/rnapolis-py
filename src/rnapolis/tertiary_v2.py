import string
from functools import cached_property
from typing import List, Optional

import numpy as np
import pandas as pd

# Constants
AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT = 1.6


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
            grouped = self.atoms.groupby(groupby_cols, dropna=False, observed=False)

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

    @cached_property
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

    @cached_property
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

    @cached_property
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
        if self.format == "PDB":
            mask = self.atoms["name"] == atom_name
            atoms_df = self.atoms[mask]
            if len(atoms_df) > 0:
                return Atom(atoms_df.iloc[0], self.format)
        elif self.format == "mmCIF":
            if "auth_atom_id" in self.atoms.columns:
                mask = self.atoms["auth_atom_id"] == atom_name
                atoms_df = self.atoms[mask]
                if len(atoms_df) > 0:
                    return Atom(atoms_df.iloc[0], self.format)
            else:
                mask = self.atoms["label_atom_id"] == atom_name
                atoms_df = self.atoms[mask]
                if len(atoms_df) > 0:
                    return Atom(atoms_df.iloc[0], self.format)
        return None

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
        if self.chain_id.isspace() or not self.chain_id:
            builder = f"{self.residue_name}"
        else:
            builder = f"{self.chain_id}.{self.residue_name}"

        # Add a separator if the residue name ends with a digit
        if len(self.residue_name) > 0 and self.residue_name[-1] in string.digits:
            builder += "/"

        # Add residue number
        builder += f"{self.residue_number}"

        # Add insertion code if present
        if self.insertion_code is not None:
            builder += f"^{self.insertion_code}"

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
