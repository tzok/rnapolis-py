from functools import cached_property
from typing import List, Optional

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
        self.format = atoms.attrs.get('format', 'unknown')
    
    @cached_property
    def residues(self) -> List[pd.DataFrame]:
        """
        Group atoms by residue and return a list of DataFrames, each representing a residue.
        
        The grouping logic depends on the format of the input data:
        - For PDB: group by (chainID, resSeq, iCode)
        - For mmCIF: group by (label_asym_id, label_seq_id) if present, 
                     otherwise by (auth_asym_id, auth_seq_id, pdbx_PDB_ins_code)
        
        Returns:
        --------
        List[pd.DataFrame]
            List of DataFrames, each containing atoms for a single residue
        """
        if self.format == 'PDB':
            # Group by chain ID, residue sequence number, and insertion code
            groupby_cols = ['chainID', 'resSeq', 'iCode']
            
            # Filter out columns that don't exist in the DataFrame
            groupby_cols = [col for col in groupby_cols if col in self.atoms.columns]
            
            # Group atoms by residue
            grouped = self.atoms.groupby(groupby_cols, dropna=False)
            
        elif self.format == 'mmCIF':
            # Prefer label_* columns if they exist
            if 'label_asym_id' in self.atoms.columns and 'label_seq_id' in self.atoms.columns:
                groupby_cols = ['label_asym_id', 'label_seq_id']
                
                # Add insertion code if it exists
                if 'pdbx_PDB_ins_code' in self.atoms.columns:
                    groupby_cols.append('pdbx_PDB_ins_code')
            else:
                # Fall back to auth_* columns
                groupby_cols = ['auth_asym_id', 'auth_seq_id']
                
                # Add insertion code if it exists
                if 'pdbx_PDB_ins_code' in self.atoms.columns:
                    groupby_cols.append('pdbx_PDB_ins_code')
            
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
            residue_df.attrs['format'] = self.format
            
            residue_dfs.append(residue_df)
        
        return residue_dfs
