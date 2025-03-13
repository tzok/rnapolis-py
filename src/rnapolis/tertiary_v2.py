from functools import cached_property
from typing import List, Optional
import string

import numpy as np
import pandas as pd


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

    @property
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

    @property
    def element(self) -> str:
        """Get the element symbol."""
        if self.format == "PDB":
            return self.data["element"]
        elif self.format == "mmCIF":
            if "type_symbol" in self.data:
                return self.data["type_symbol"]
        return ""

    @property
    def coordinates(self) -> np.ndarray:
        """Get the 3D coordinates of the atom."""
        if self.format == "PDB":
            return np.array([self.data["x"], self.data["y"], self.data["z"]])
        elif self.format == "mmCIF":
            return np.array(
                [self.data["Cartn_x"], self.data["Cartn_y"], self.data["Cartn_z"]]
            )
        return np.array([0.0, 0.0, 0.0])

    @property
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

    @property
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
            if "auth_comp_id" in self.atoms.columns:
                return self.atoms["auth_comp_id"].iloc[0]
            else:
                return self.atoms["label_comp_id"].iloc[0]
        return ""

    @cached_property
    def atoms_list(self) -> List[Atom]:
        """Get a list of all atoms in this residue."""
        return [Atom(self.atoms.iloc[i], self.format) for i in range(len(self.atoms))]

    def find_atom(self, atom_name: str) -> Optional[Atom]:
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
        if self.insertion_code:
            builder += f"^{self.insertion_code}"

        return builder

    def __repr__(self) -> str:
        """Detailed string representation of the residue."""
        return f"Residue({self.__str__()}, {len(self.atoms)} atoms)"
