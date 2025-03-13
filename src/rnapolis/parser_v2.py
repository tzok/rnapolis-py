from typing import IO, Union

import pandas as pd
from mmcif.io.IoAdapterPy import IoAdapterPy


def parse_pdb_atoms(content: Union[str, IO[str]]) -> pd.DataFrame:
    """
    Parse PDB file content and extract ATOM and HETATM records into a pandas DataFrame.

    Parameters:
    -----------
    content : Union[str, IO[str]]
        Content of a PDB file as a string or file-like object

    Returns:
    --------
    pd.DataFrame
        DataFrame containing parsed ATOM and HETATM records with columns corresponding to PDB format
    """
    records = []

    # Handle both string content and file-like objects
    if isinstance(content, str):
        lines = content.splitlines()
    else:
        # Read all lines from the file-like object
        content.seek(0)  # Ensure we're at the beginning of the file
        lines = content.readlines()
        # Convert bytes to string if needed
        if isinstance(lines[0], bytes):
            lines = [line.decode("utf-8") for line in lines]

    for line in lines:
        record_type = line[:6].strip()

        # Only process ATOM and HETATM records
        if record_type not in ["ATOM", "HETATM"]:
            continue

        # Parse fields according to PDB format specification
        icode = line[26:27].strip()
        record = {
            "record_type": record_type,
            "serial": line[6:11].strip(),
            "name": line[12:16].strip(),
            "altLoc": line[16:17].strip(),
            "resName": line[17:20].strip(),
            "chainID": line[21:22].strip(),
            "resSeq": line[22:26].strip(),
            "iCode": None if not icode else icode,  # Convert empty string to None
            "x": line[30:38].strip(),
            "y": line[38:46].strip(),
            "z": line[46:54].strip(),
            "occupancy": line[54:60].strip(),
            "tempFactor": line[60:66].strip(),
            "element": line[76:78].strip(),
            "charge": line[78:80].strip(),
        }

        records.append(record)

    # Create DataFrame from records
    if not records:
        # Return empty DataFrame with correct columns if no records found
        return pd.DataFrame(
            columns=[
                "record_type",
                "serial",
                "name",
                "altLoc",
                "resName",
                "chainID",
                "resSeq",
                "iCode",
                "x",
                "y",
                "z",
                "occupancy",
                "tempFactor",
                "element",
                "charge",
            ]
        )

    df = pd.DataFrame(records)

    # Convert numeric columns to appropriate types
    numeric_columns = ["serial", "resSeq", "x", "y", "z", "occupancy", "tempFactor"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert categorical columns
    categorical_columns = [
        "record_type",
        "name",
        "altLoc",
        "resName",
        "chainID",
        "element",
        "charge",
    ]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Add format attribute to the DataFrame
    df.attrs["format"] = "PDB"

    return df


def parse_cif_atoms(content: Union[str, IO[str]]) -> pd.DataFrame:
    """
    Parse mmCIF file content and extract atom_site records into a pandas DataFrame.

    Parameters:
    -----------
    content : Union[str, IO[str]]
        Content of a mmCIF file as a string or file-like object

    Returns:
    --------
    pd.DataFrame
        DataFrame containing parsed atom_site records with columns corresponding to mmCIF format
    """
    adapter = IoAdapterPy()

    # Handle both string content and file-like objects
    if isinstance(content, str):
        # Create a temporary file to use with the adapter
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".cif") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            data = adapter.readFile(temp_file.name)
    else:
        # Assume it's a file-like object with a name attribute
        data = adapter.readFile(content.name)

    # Get the atom_site category
    category = data[0].getObj("atom_site")

    if not category:
        # Return empty DataFrame if no atom_site category found
        return pd.DataFrame()

    # Extract attribute names and data rows
    attributes = category.getAttributeList()
    rows = category.getRowList()

    # Create a list of dictionaries for each atom
    records = []
    for row in rows:
        record = dict(zip(attributes, row))

        # Convert "?" or "." in insertion code to None
        if "pdbx_PDB_ins_code" in record:
            if record["pdbx_PDB_ins_code"] in ["?", ".", ""]:
                record["pdbx_PDB_ins_code"] = None

        records.append(record)

    # Create DataFrame from records
    df = pd.DataFrame(records)

    # Convert numeric columns to appropriate types
    numeric_columns = [
        "id",
        "auth_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "occupancy",
        "B_iso_or_equiv",
        "pdbx_formal_charge",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert categorical columns
    categorical_columns = [
        "group_PDB",
        "type_symbol",
        "label_atom_id",
        "label_comp_id",
        "label_asym_id",
        "auth_atom_id",
        "auth_comp_id",
        "auth_asym_id",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Add format attribute to the DataFrame
    df.attrs["format"] = "mmCIF"

    return df
