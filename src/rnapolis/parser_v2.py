import pandas as pd
from typing import List, Dict, Any


def parse_pdb(content: str) -> pd.DataFrame:
    """
    Parse PDB file content and extract ATOM and HETATM records into a pandas DataFrame.
    
    Parameters:
    -----------
    content : str
        Content of a PDB file as a string
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing parsed ATOM and HETATM records with columns corresponding to PDB format
    """
    records = []
    
    for line in content.splitlines():
        record_type = line[:6].strip()
        
        # Only process ATOM and HETATM records
        if record_type not in ["ATOM", "HETATM"]:
            continue
            
        # Parse fields according to PDB format specification
        record = {
            "record_type": record_type,
            "serial": line[6:11].strip(),
            "name": line[12:16].strip(),
            "altLoc": line[16:17].strip(),
            "resName": line[17:20].strip(),
            "chainID": line[21:22].strip(),
            "resSeq": line[22:26].strip(),
            "iCode": line[26:27].strip(),
            "x": line[30:38].strip(),
            "y": line[38:46].strip(),
            "z": line[46:54].strip(),
            "occupancy": line[54:60].strip(),
            "tempFactor": line[60:66].strip(),
            "element": line[76:78].strip(),
            "charge": line[78:80].strip()
        }
        
        records.append(record)
    
    # Create DataFrame from records
    if not records:
        # Return empty DataFrame with correct columns if no records found
        return pd.DataFrame(columns=[
            "record_type", "serial", "name", "altLoc", "resName", "chainID", 
            "resSeq", "iCode", "x", "y", "z", "occupancy", "tempFactor", 
            "element", "charge"
        ])
    
    df = pd.DataFrame(records)
    
    # Convert numeric columns to appropriate types
    numeric_columns = ["serial", "resSeq", "x", "y", "z", "occupancy", "tempFactor"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert categorical columns
    categorical_columns = ["record_type", "name", "resName", "chainID", "element"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")
    
    return df
