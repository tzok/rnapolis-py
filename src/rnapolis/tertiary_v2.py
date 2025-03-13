from functools import cached_property
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np


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
            grouped = self.atoms.groupby(groupby_cols, dropna=False)

        elif self.format == "mmCIF":
            # Prefer label_* columns if they exist
            if (
                "label_asym_id" in self.atoms.columns
                and "label_seq_id" in self.atoms.columns
            ):
                groupby_cols = ["label_asym_id", "label_seq_id"]

                # Add insertion code if it exists
                if "pdbx_PDB_ins_code" in self.atoms.columns:
                    groupby_cols.append("pdbx_PDB_ins_code")
            else:
                # Fall back to auth_* columns
                groupby_cols = ["auth_asym_id", "auth_seq_id"]

                # Add insertion code if it exists
                if "pdbx_PDB_ins_code" in self.atoms.columns:
                    groupby_cols.append("pdbx_PDB_ins_code")

            # Group atoms by residue
            grouped = self.atoms.groupby(groupby_cols, dropna=False)

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
            if "label_asym_id" in self.atoms.columns:
                return self.atoms["label_asym_id"].iloc[0]
            else:
                return self.atoms["auth_asym_id"].iloc[0]
        return ""

    @cached_property
    def residue_number(self) -> int:
        """Get the residue sequence number."""
        if self.format == "PDB":
            return int(self.atoms["resSeq"].iloc[0])
        elif self.format == "mmCIF":
            if "label_seq_id" in self.atoms.columns:
                return int(self.atoms["label_seq_id"].iloc[0])
            else:
                return int(self.atoms["auth_seq_id"].iloc[0])
        return 0

    @cached_property
    def insertion_code(self) -> str:
        """Get the insertion code, if any."""
        if self.format == "PDB":
            icode = self.atoms["iCode"].iloc[0]
            return icode if pd.notna(icode) else ""
        elif self.format == "mmCIF":
            if "pdbx_PDB_ins_code" in self.atoms.columns:
                icode = self.atoms["pdbx_PDB_ins_code"].iloc[0]
                return icode if pd.notna(icode) else ""
        return ""

    @cached_property
    def residue_name(self) -> str:
        """Get the residue name (e.g., 'A', 'G', 'C', 'U', etc.)."""
        if self.format == "PDB":
            return self.atoms["resName"].iloc[0]
        elif self.format == "mmCIF":
            if "label_comp_id" in self.atoms.columns:
                return self.atoms["label_comp_id"].iloc[0]
            else:
                return self.atoms["auth_comp_id"].iloc[0]
        return ""

    @cached_property
    def center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the residue."""
        if self.format == "PDB":
            coords = self.atoms[["x", "y", "z"]].values
            return np.mean(coords, axis=0)
        elif self.format == "mmCIF":
            coords = self.atoms[["Cartn_x", "Cartn_y", "Cartn_z"]].values
            return np.mean(coords, axis=0)
        return np.array([0.0, 0.0, 0.0])

    def find_atom(self, atom_name: str) -> Optional[pd.Series]:
        """
        Find an atom by name in this residue.

        Parameters:
        -----------
        atom_name : str
            Name of the atom to find

        Returns:
        --------
        Optional[pd.Series]
            The atom data as a pandas Series, or None if not found
        """
        if self.format == "PDB":
            mask = self.atoms["name"] == atom_name
            atoms = self.atoms[mask]
            if len(atoms) > 0:
                return atoms.iloc[0]
        elif self.format == "mmCIF":
            if "label_atom_id" in self.atoms.columns:
                mask = self.atoms["label_atom_id"] == atom_name
                atoms = self.atoms[mask]
                if len(atoms) > 0:
                    return atoms.iloc[0]
            else:
                mask = self.atoms["auth_atom_id"] == atom_name
                atoms = self.atoms[mask]
                if len(atoms) > 0:
                    return atoms.iloc[0]
        return None

    def get_coordinates(self, atom_name: str) -> Optional[np.ndarray]:
        """
        Get the coordinates of a specific atom.

        Parameters:
        -----------
        atom_name : str
            Name of the atom

        Returns:
        --------
        Optional[np.ndarray]
            3D coordinates as a numpy array, or None if atom not found
        """
        atom = self.find_atom(atom_name)
        if atom is not None:
            if self.format == "PDB":
                return np.array([atom["x"], atom["y"], atom["z"]])
            elif self.format == "mmCIF":
                return np.array([atom["Cartn_x"], atom["Cartn_y"], atom["Cartn_z"]])
        return None

    def __str__(self) -> str:
        """String representation of the residue."""
        return f"{self.residue_name} {self.chain_id}:{self.residue_number}{self.insertion_code}"

    def __repr__(self) -> str:
        """Detailed string representation of the residue."""
        return f"Residue({self.__str__()}, {len(self.atoms)} atoms)"
