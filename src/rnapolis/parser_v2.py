import io
import os
import string
import tempfile
from typing import IO, TextIO, Union

import pandas as pd
from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory, DataContainer


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

    current_model = 1
    for line in lines:
        record_type = line[:6].strip()

        # Check for MODEL record
        if record_type == "MODEL":
            try:
                current_model = int(line[10:14].strip())
            except ValueError:
                # Handle cases where MODEL record might be malformed
                pass  # Keep the previous model number
            continue

        # Only process ATOM and HETATM records
        if record_type not in ["ATOM", "HETATM"]:
            continue

        # Parse fields according to PDB format specification
        alt_loc = line[16:17].strip()
        icode = line[26:27].strip()
        element = line[76:78].strip()
        charge = line[78:80].strip()

        record = {
            "record_type": record_type,
            "serial": line[6:11].strip(),
            "name": line[12:16].strip(),
            "altLoc": None if not alt_loc else alt_loc,  # Store None if empty
            "resName": line[17:20].strip(),
            "chainID": line[21:22].strip(),
            "resSeq": line[22:26].strip(),
            "iCode": None if not icode else icode,  # Store None if empty
            "x": line[30:38].strip(),
            "y": line[38:46].strip(),
            "z": line[46:54].strip(),
            "occupancy": line[54:60].strip(),
            "tempFactor": line[60:66].strip(),
            "element": None if not element else element,  # Store None if empty
            "charge": None if not charge else charge,  # Store None if empty
            "model": current_model,  # Add the current model number
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
                "model",
            ]
        )

    df = pd.DataFrame(records)

    # Convert numeric columns to appropriate types
    numeric_columns = [
        "serial",
        "resSeq",
        "x",
        "y",
        "z",
        "occupancy",
        "tempFactor",
        "model",
    ]
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

    # Handle string, StringIO, and file-like objects
    if isinstance(content, str):
        # Create a temporary file for string input
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".cif", delete=False
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        try:
            data = adapter.readFile(temp_file_path)
        finally:
            os.remove(temp_file_path)  # Clean up the temporary file
    elif isinstance(content, io.StringIO):
        # Create a temporary file for StringIO input
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".cif", delete=False
        ) as temp_file:
            content.seek(0)  # Ensure reading from the start
            temp_file.write(content.read())
            temp_file_path = temp_file.name
        try:
            data = adapter.readFile(temp_file_path)
        finally:
            os.remove(temp_file_path)  # Clean up the temporary file
    elif hasattr(content, "name"):
        # Assume it's a file-like object with a name attribute (like an open file)
        data = adapter.readFile(content.name)
    else:
        raise TypeError(
            "Unsupported input type for parse_cif_atoms. Expected str, file-like object with name, or StringIO."
        )

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
        record = {}
        for attr, value in zip(attributes, row):
            # Store None if value indicates missing data ('?' or '.')
            if value in ["?", "."]:
                record[attr] = None
            else:
                record[attr] = value
        records.append(record)

    # Create DataFrame from records
    df = pd.DataFrame(records)

    # Define columns based on mmCIF specification for atom_site
    float_cols = [
        "aniso_B[1][1]",
        "aniso_B[1][1]_esd",
        "aniso_B[1][2]",
        "aniso_B[1][2]_esd",
        "aniso_B[1][3]",
        "aniso_B[1][3]_esd",
        "aniso_B[2][2]",
        "aniso_B[2][2]_esd",
        "aniso_B[2][3]",
        "aniso_B[2][3]_esd",
        "aniso_B[3][3]",
        "aniso_B[3][3]_esd",
        "aniso_ratio",
        "aniso_U[1][1]",
        "aniso_U[1][1]_esd",
        "aniso_U[1][2]",
        "aniso_U[1][2]_esd",
        "aniso_U[1][3]",
        "aniso_U[1][3]_esd",
        "aniso_U[2][2]",
        "aniso_U[2][2]_esd",
        "aniso_U[2][3]",
        "aniso_U[2][3]_esd",
        "aniso_U[3][3]",
        "aniso_U[3][3]_esd",
        "B_equiv_geom_mean",
        "B_equiv_geom_mean_esd",
        "B_iso_or_equiv",
        "B_iso_or_equiv_esd",
        "Cartn_x",
        "Cartn_x_esd",
        "Cartn_y",
        "Cartn_y_esd",
        "Cartn_z",
        "Cartn_z_esd",
        "fract_x",
        "fract_x_esd",
        "fract_y",
        "fract_y_esd",
        "fract_z",
        "fract_z_esd",
        "occupancy",
        "occupancy_esd",
        "U_equiv_geom_mean",
        "U_equiv_geom_mean_esd",
        "U_iso_or_equiv",
        "U_iso_or_equiv_esd",
    ]
    int_cols = [
        "attached_hydrogens",
        "label_seq_id",
        "symmetry_multiplicity",
        "pdbx_PDB_model_num",
        "pdbx_formal_charge",
        "pdbx_label_index",
    ]
    category_cols = [
        "auth_asym_id",
        "auth_atom_id",
        "auth_comp_id",
        "auth_seq_id",
        "calc_attached_atom",
        "calc_flag",
        "disorder_assembly",
        "disorder_group",
        "group_PDB",
        "id",
        "label_alt_id",
        "label_asym_id",
        "label_atom_id",
        "label_comp_id",
        "label_entity_id",
        "thermal_displace_type",
        "type_symbol",
        "pdbx_atom_ambiguity",
        "adp_type",
        "refinement_flags",
        "refinement_flags_adp",
        "refinement_flags_occupancy",
        "refinement_flags_posn",
        "pdbx_auth_alt_id",
        "pdbx_PDB_ins_code",
        "pdbx_PDB_residue_no",
        "pdbx_PDB_residue_name",
        "pdbx_PDB_strand_id",
        "pdbx_PDB_atom_name",
        "pdbx_auth_atom_name",
        "pdbx_auth_comp_id",
        "pdbx_auth_asym_id",
        "pdbx_auth_seq_id",
        "pdbx_tls_group_id",
        "pdbx_ncs_dom_id",
        "pdbx_group_NDB",
        "pdbx_atom_group",
        "pdbx_label_seq_num",
        "pdbx_not_in_asym",
        "pdbx_sifts_xref_db_name",
        "pdbx_sifts_xref_db_acc",
        "pdbx_sifts_xref_db_num",
        "pdbx_sifts_xref_db_res",
    ]

    # Convert columns to appropriate types
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in int_cols:
        if col in df.columns:
            # Use Int64 (nullable integer) to handle potential NaNs from coercion
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Add format attribute to the DataFrame
    df.attrs["format"] = "mmCIF"

    return df


def can_write_pdb(df: pd.DataFrame) -> bool:
    """
    Check if the DataFrame can be losslessly represented in PDB format.

    PDB format has limitations on field widths:
    - Atom serial number (id): max 99999
    - Chain identifier (auth_asym_id): max 1 character
    - Residue sequence number (auth_seq_id): max 9999

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing atom records, as created by parse_pdb_atoms or parse_cif_atoms.

    Returns:
    --------
    bool
        True if the DataFrame can be written to PDB format without data loss/truncation, False otherwise.
    """
    format_type = df.attrs.get("format")

    if format_type == "PDB":
        # Assume data originally from PDB already fits PDB constraints
        return True

    if df.empty:
        # An empty DataFrame can be represented as an empty PDB file
        return True

    if format_type == "mmCIF":
        # Check serial number (id)
        # Convert to numeric first to handle potential categorical type and NaNs
        if "id" not in df.columns or (
            pd.to_numeric(df["id"], errors="coerce").max() > 99999
        ):
            return False

        # Check chain ID (auth_asym_id) length
        if "auth_asym_id" not in df.columns or (
            df["auth_asym_id"].dropna().astype(str).str.len().max() > 1
        ):
            return False

        # Check residue sequence number (auth_seq_id)
        if "auth_seq_id" not in df.columns or (
            pd.to_numeric(df["auth_seq_id"], errors="coerce").max() > 9999
        ):
            return False

        # All checks passed for mmCIF
        return True

    # If format is unknown or not PDB/mmCIF, assume it cannot be safely written
    return False


def fit_to_pdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to fit the atom data in a DataFrame to comply with PDB format limitations.

    If the data already fits (checked by can_write_pdb), returns the original DataFrame.
    Otherwise, checks if fitting is possible based on total atoms, unique chains,
    and residues per chain. If fitting is possible, it renumbers atoms, renames chains,
    and renumbers residues within each chain sequentially starting from 1.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing atom records, as created by parse_pdb_atoms or parse_cif_atoms.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with data potentially modified to fit PDB constraints.
        The 'format' attribute of the returned DataFrame will be set to 'PDB'.

    Raises:
    -------
    ValueError
        If the data cannot be fitted into PDB format constraints (too many atoms,
        chains, or residues per chain).
    """
    format_type = df.attrs.get("format")

    if not format_type:
        raise ValueError("DataFrame format attribute is not set.")

    if can_write_pdb(df):
        return df

    # Determine column names based on format
    if format_type == "PDB":
        serial_col = "serial"
        chain_col = "chainID"
        resseq_col = "resSeq"
        icode_col = "iCode"
    elif format_type == "mmCIF":
        serial_col = "id"
        chain_col = "auth_asym_id"
        resseq_col = "auth_seq_id"
        icode_col = "pdbx_PDB_ins_code"
    else:
        raise ValueError(f"Unsupported DataFrame format: {format_type}")

    # --- Feasibility Checks ---
    if chain_col not in df.columns:
        raise ValueError(f"Missing required chain column: {chain_col}")
    if resseq_col not in df.columns:
        raise ValueError(f"Missing required residue sequence column: {resseq_col}")

    unique_chains = df[chain_col].unique()
    num_chains = len(unique_chains)
    total_atoms = len(df)
    max_pdb_serial = 99999
    max_pdb_residue = 9999
    available_chain_ids = list(
        string.ascii_uppercase + string.ascii_lowercase + string.digits
    )
    max_pdb_chains = len(available_chain_ids)

    # Check 1: Total atoms + TER lines <= 99999
    if total_atoms + num_chains > max_pdb_serial:
        raise ValueError(
            f"Cannot fit to PDB: Total atoms ({total_atoms}) + TER lines ({num_chains}) exceeds PDB limit ({max_pdb_serial})."
        )

    # Check 2: Number of chains <= 62
    if num_chains > max_pdb_chains:
        raise ValueError(
            f"Cannot fit to PDB: Number of unique chains ({num_chains}) exceeds PDB limit ({max_pdb_chains})."
        )

    # Check 3: Max residues per chain <= 9999
    # More accurate check: group by chain, then count unique (resSeq, iCode) tuples
    # Use a temporary structure to avoid modifying the original df
    check_df = pd.DataFrame(
        {
            "chain": df[chain_col],
            "resSeq": df[resseq_col],
            "iCode": df[icode_col].fillna("") if icode_col in df.columns else "",
        }
    )
    residue_counts = check_df.groupby("chain").apply(
        lambda x: x[["resSeq", "iCode"]].drop_duplicates().shape[0]
    )
    max_residues_per_chain = residue_counts.max() if not residue_counts.empty else 0

    if max_residues_per_chain > max_pdb_residue:
        raise ValueError(
            f"Cannot fit to PDB: Maximum residues in a single chain ({max_residues_per_chain}) exceeds PDB limit ({max_pdb_residue})."
        )

    # --- Perform Fitting ---
    df_fitted = df.copy()

    # 1. Rename Chains
    chain_mapping = {
        orig_chain: available_chain_ids[i] for i, orig_chain in enumerate(unique_chains)
    }
    df_fitted[chain_col] = df_fitted[chain_col].map(chain_mapping)
    # Ensure the chain column is treated as string/object after mapping
    df_fitted[chain_col] = df_fitted[chain_col].astype(object)

    # 2. Renumber Residues within each new chain
    new_resseq_col = "new_resSeq"  # Temporary column for new numbering
    df_fitted[new_resseq_col] = -1  # Initialize

    all_new_res_maps = {}
    for new_chain_id, group in df_fitted.groupby(chain_col):
        # Identify unique original residues (seq + icode) in order of appearance
        original_residues = group[[resseq_col, icode_col]].drop_duplicates()
        # Create mapping: (orig_resSeq, orig_iCode) -> new_resSeq (1-based)
        residue_mapping = {
            tuple(res): i + 1
            for i, res in enumerate(original_residues.itertuples(index=False))
        }
        all_new_res_maps[new_chain_id] = residue_mapping

        # Apply mapping to the group
        res_indices = group.set_index([resseq_col, icode_col]).index
        df_fitted.loc[group.index, new_resseq_col] = res_indices.map(residue_mapping)

    # Replace original residue number and clear insertion code
    df_fitted[resseq_col] = df_fitted[new_resseq_col]
    df_fitted[icode_col] = None  # Insertion codes are now redundant
    df_fitted.drop(columns=[new_resseq_col], inplace=True)
    # Convert resseq_col back to Int64 if it was before, handling potential NaNs if any step failed
    df_fitted[resseq_col] = df_fitted[resseq_col].astype("Int64")

    # 3. Renumber Atom Serials
    new_serial_col = "new_serial"
    df_fitted[new_serial_col] = -1  # Initialize
    current_serial = 0
    last_chain_id_for_serial = None

    # Iterate in the potentially re-sorted order after grouping/mapping
    # Ensure stable sort order for consistent serial numbering
    df_fitted.sort_index(
        inplace=True
    )  # Sort by original index to maintain original atom order as much as possible

    for index, row in df_fitted.iterrows():
        current_chain_id = row[chain_col]
        if (
            last_chain_id_for_serial is not None
            and current_chain_id != last_chain_id_for_serial
        ):
            current_serial += 1  # Increment for TER line

        current_serial += 1
        if current_serial > max_pdb_serial:
            # This should have been caught by the initial check, but is a safeguard
            raise ValueError("Serial number exceeded PDB limit during renumbering.")

        df_fitted.loc[index, new_serial_col] = current_serial
        last_chain_id_for_serial = current_chain_id

    # Replace original serial number
    df_fitted[serial_col] = df_fitted[new_serial_col]
    df_fitted.drop(columns=[new_serial_col], inplace=True)
    # Convert serial_col back to Int64
    df_fitted[serial_col] = df_fitted[serial_col].astype("Int64")

    # Update attributes and column types for PDB compatibility
    df_fitted.attrs["format"] = "PDB"

    # Ensure final column types match expected PDB output (especially categories)
    # Reapply categorical conversion as some operations might change dtypes
    pdb_categorical_cols = [
        "record_type",
        "name",
        "altLoc",
        "resName",
        chain_col,
        "element",
        "charge",
        icode_col,
    ]
    if "record_type" not in df_fitted.columns and "group_PDB" in df_fitted.columns:
        df_fitted.rename(
            columns={"group_PDB": "record_type"}, inplace=True
        )  # Ensure correct name

    for col in pdb_categorical_cols:
        if col in df_fitted.columns:
            # Handle None explicitly before converting to category if needed
            if df_fitted[col].isnull().any():
                df_fitted[col] = (
                    df_fitted[col].astype(object).fillna("")
                )  # Fill None with empty string for category
            df_fitted[col] = df_fitted[col].astype("category")

    # Rename columns if necessary from mmCIF to PDB standard names
    rename_map = {
        "id": "serial",
        "auth_asym_id": "chainID",
        "auth_seq_id": "resSeq",
        "pdbx_PDB_ins_code": "iCode",
        "label_atom_id": "name",  # Prefer label_atom_id if auth_atom_id not present? PDB uses 'name'
        "label_comp_id": "resName",  # Prefer label_comp_id if auth_comp_id not present? PDB uses 'resName'
        "type_symbol": "element",
        "pdbx_formal_charge": "charge",
        "Cartn_x": "x",
        "Cartn_y": "y",
        "Cartn_z": "z",
        "B_iso_or_equiv": "tempFactor",
        "group_PDB": "record_type",
        "pdbx_PDB_model_num": "model",
        # Add mappings for auth_atom_id -> name, auth_comp_id -> resName if needed,
        # deciding on precedence if both label_* and auth_* exist.
        # Current write_pdb prioritizes auth_* when reading mmCIF, so map those.
        "auth_atom_id": "name",
        "auth_comp_id": "resName",
    }

    # Only rename columns that actually exist in the DataFrame
    actual_rename_map = {k: v for k, v in rename_map.items() if k in df_fitted.columns}
    df_fitted.rename(columns=actual_rename_map, inplace=True)

    # Ensure essential PDB columns exist, even if empty, if they were created during fitting
    pdb_essential_cols = [
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
        "model",
    ]
    for col in pdb_essential_cols:
        if col not in df_fitted.columns:
            # This case might occur if input mmCIF was missing fundamental columns mapped to PDB essentials
            # Decide on default value or raise error. Adding empty series for now.
            df_fitted[col] = pd.Series(
                dtype="object"
            )  # Add as object to handle potential None/mixed types initially

    # Re-order columns to standard PDB order for clarity
    final_pdb_order = [col for col in pdb_essential_cols if col in df_fitted.columns]
    other_cols = [col for col in df_fitted.columns if col not in final_pdb_order]
    df_fitted = df_fitted[final_pdb_order + other_cols]

    # --- Final Type Conversions for PDB format ---
    # Convert numeric columns (similar to parse_pdb_atoms)
    pdb_numeric_columns = [
        "serial",
        "resSeq",
        "x",
        "y",
        "z",
        "occupancy",
        "tempFactor",
        "model",
    ]
    for col in pdb_numeric_columns:
        if col in df_fitted.columns:
            # Use Int64 for integer-like columns that might have been NaN during processing
            if col in ["serial", "resSeq", "model"]:
                df_fitted[col] = pd.to_numeric(df_fitted[col], errors="coerce").astype(
                    "Int64"
                )
            else:  # Floats
                df_fitted[col] = pd.to_numeric(df_fitted[col], errors="coerce")

    # Convert categorical columns (similar to parse_pdb_atoms)
    # Note: chainID and iCode were already handled during fitting/renaming
    pdb_categorical_columns_final = [
        "record_type",
        "name",
        "altLoc",
        "resName",
        "chainID",  # Already category, but ensure consistency
        "iCode",  # Already category, but ensure consistency
        "element",
        "charge",
    ]
    for col in pdb_categorical_columns_final:
        if col in df_fitted.columns:
            # Ensure the column is categorical first
            if not pd.api.types.is_categorical_dtype(df_fitted[col]):
                # Convert non-categorical columns, handling potential NaNs
                if df_fitted[col].isnull().any():
                    df_fitted[col] = (
                        df_fitted[col].astype(object).fillna("").astype("category")
                    )
                else:
                    df_fitted[col] = df_fitted[col].astype("category")
            else:
                # If already categorical, check if '' needs to be added before fillna
                has_nans = df_fitted[col].isnull().any()
                if has_nans and "" not in df_fitted[col].cat.categories:
                    # Add '' category explicitly
                    df_fitted[col] = df_fitted[col].cat.add_categories([""])

                # Fill None/NaN with empty string (now safe)
                if has_nans:
                    df_fitted[col].fillna("", inplace=True)

    return df_fitted


def _format_pdb_atom_line(atom_data: dict) -> str:
    """Formats a dictionary of atom data into a PDB ATOM/HETATM line."""
    # PDB format specification:
    # COLUMNS        DATA TYPE     FIELD         DEFINITION
    # -----------------------------------------------------------------------
    #  1 -  6        Record name   "ATOM  " or "HETATM"
    #  7 - 11        Integer       serial        Atom serial number.
    # 13 - 16        Atom          name          Atom name.
    # 17             Character     altLoc        Alternate location indicator.
    # 18 - 20        Residue name  resName       Residue name.
    # 22             Character     chainID       Chain identifier.
    # 23 - 26        Integer       resSeq        Residue sequence number.
    # 27             AChar         iCode         Code for insertion of residues.
    # 31 - 38        Real(8.3)     x             Orthogonal coordinates for X.
    # 39 - 46        Real(8.3)     y             Orthogonal coordinates for Y.
    # 47 - 54        Real(8.3)     z             Orthogonal coordinates for Z.
    # 55 - 60        Real(6.2)     occupancy     Occupancy.
    # 61 - 66        Real(6.2)     tempFactor    Temperature factor.
    # 77 - 78        LString(2)    element       Element symbol, right-justified.
    # 79 - 80        LString(2)    charge        Charge on the atom.

    # Record name (ATOM/HETATM)
    record_name = atom_data.get("record_name", "ATOM").ljust(6)

    # Serial number
    serial = str(atom_data.get("serial", 0)).rjust(5)

    # Atom name - special alignment rules
    atom_name = atom_data.get("name", "")
    if len(atom_name) < 4 and atom_name[:1].isalpha():
        # Pad with space on left for 1-3 char names starting with a letter
        atom_name_fmt = (" " + atom_name).ljust(4)
    else:
        # Use as is, left-justified, for 4-char names or those starting with a digit
        atom_name_fmt = atom_name.ljust(4)

    # Alternate location indicator
    alt_loc = atom_data.get("altLoc", "")[:1].ljust(1)  # Max 1 char

    # Residue name
    res_name = atom_data.get("resName", "").rjust(
        3
    )  # Spec says "Residue name", examples often right-justified

    # Chain identifier
    chain_id = atom_data.get("chainID", "")[:1].ljust(1)  # Max 1 char

    # Residue sequence number
    res_seq = str(atom_data.get("resSeq", 0)).rjust(4)

    # Insertion code
    icode = atom_data.get("iCode", "")[:1].ljust(1)  # Max 1 char

    # Coordinates
    x = f"{atom_data.get('x', 0.0):8.3f}"
    y = f"{atom_data.get('y', 0.0):8.3f}"
    z = f"{atom_data.get('z', 0.0):8.3f}"

    # Occupancy
    occupancy = f"{atom_data.get('occupancy', 1.0):6.2f}"

    # Temperature factor
    temp_factor = f"{atom_data.get('tempFactor', 0.0):6.2f}"

    # Element symbol
    element = atom_data.get("element", "").rjust(2)

    # Charge
    charge_val = atom_data.get("charge", "")
    charge_fmt = ""
    if charge_val:
        try:
            # Try converting numeric charge (e.g., +1, -2) to PDB format (1+, 2-)
            charge_int = int(float(charge_val))  # Use float first for cases like "1.0"
            if charge_int != 0:
                charge_fmt = f"{abs(charge_int)}{'+' if charge_int > 0 else '-'}"
        except ValueError:
            # If already formatted (e.g., "1+", "FE2+"), use its string representation
            charge_fmt = str(charge_val)
        # Ensure it fits and is right-justified
        charge_fmt = charge_fmt.strip()[:2].rjust(2)
    else:
        charge_fmt = "  "  # Blank if no charge

    # Construct the full line
    # Ensure spacing is correct according to the spec
    # 1-6 Record name | 7-11 Serial | 12 Space | 13-16 Name | 17 AltLoc | 18-20 ResName | 21 Space | 22 ChainID | 23-26 ResSeq | 27 iCode | 28-30 Spaces | 31-38 X | 39-46 Y | 47-54 Z | 55-60 Occupancy | 61-66 TempFactor | 67-76 Spaces | 77-78 Element | 79-80 Charge
    line = (
        f"{record_name}{serial} {atom_name_fmt}{alt_loc}{res_name} {chain_id}{res_seq}{icode}   "
        f"{x}{y}{z}{occupancy}{temp_factor}          "  # 10 spaces
        f"{element}{charge_fmt}"
    )

    # Ensure the line is exactly 80 characters long
    return line.ljust(80)


def write_pdb(
    df: pd.DataFrame, output: Union[str, TextIO, None] = None
) -> Union[str, None]:
    """
    Write a DataFrame of atom records to PDB format.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing atom records, as created by parse_pdb_atoms or parse_cif_atoms.
        Must contain columns mappable to PDB format fields.
    output : Union[str, TextIO, None], optional
        Output file path or file-like object. If None, returns the PDB content as a string.

    Returns:
    --------
    Union[str, None]
        If output is None, returns the PDB content as a string. Otherwise, returns None.
    """
    buffer = io.StringIO()
    format_type = df.attrs.get("format", "PDB")  # Assume PDB if not specified

    last_model_num = None
    last_chain_id = None
    last_res_info = None  # Tuple (resSeq, iCode, resName) for TER record
    last_serial = 0

    # Check if DataFrame is empty
    if df.empty:
        buffer.write("END\n")
        content = buffer.getvalue()
        buffer.close()
        if output is not None:
            if isinstance(output, str):
                with open(output, "w") as f:
                    f.write(content)
            else:
                output.write(content)
            return None
        return content

    for _, row in df.iterrows():
        atom_data = {}

        # --- Data Extraction ---
        if format_type == "PDB":
            # Pre-process PDB values, converting None to empty strings for optional fields
            raw_alt_loc = row.get("altLoc")
            pdb_alt_loc = "" if pd.isna(raw_alt_loc) else str(raw_alt_loc)

            raw_icode = row.get("iCode")
            pdb_icode = "" if pd.isna(raw_icode) else str(raw_icode)

            raw_element = row.get("element")
            pdb_element = "" if pd.isna(raw_element) else str(raw_element)

            raw_charge = row.get("charge")
            pdb_charge = "" if pd.isna(raw_charge) else str(raw_charge)

            atom_data = {
                "record_name": row.get("record_type", "ATOM"),
                "serial": int(row.get("serial", 0)),
                "name": str(row.get("name", "")),
                "altLoc": pdb_alt_loc,
                "resName": str(row.get("resName", "")),
                "chainID": str(row.get("chainID", "")),
                "resSeq": int(row.get("resSeq", 0)),
                "iCode": pdb_icode,
                "x": float(row.get("x", 0.0)),
                "y": float(row.get("y", 0.0)),
                "z": float(row.get("z", 0.0)),
                "occupancy": float(row.get("occupancy", 1.0)),
                "tempFactor": float(row.get("tempFactor", 0.0)),
                "element": pdb_element,
                "charge": pdb_charge,
                "model": int(row.get("model", 1)),
            }
        elif format_type == "mmCIF":
            # Pre-process mmCIF values to PDB compatible format, converting None to empty strings
            raw_alt_loc = row.get("label_alt_id")
            pdb_alt_loc = "" if pd.isna(raw_alt_loc) else str(raw_alt_loc)

            raw_icode = row.get("pdbx_PDB_ins_code")
            pdb_icode = "" if pd.isna(raw_icode) else str(raw_icode)

            raw_element = row.get("type_symbol")
            pdb_element = "" if pd.isna(raw_element) else str(raw_element)

            raw_charge = row.get("pdbx_formal_charge")
            pdb_charge = "" if pd.isna(raw_charge) else str(raw_charge)

            atom_data = {
                "record_name": row.get("group_PDB", "ATOM"),
                "serial": int(row.get("id", 0)),
                "name": str(row.get("auth_atom_id", row.get("label_atom_id", ""))),
                "altLoc": pdb_alt_loc,
                "resName": str(row.get("auth_comp_id", row.get("label_comp_id", ""))),
                "chainID": str(row.get("auth_asym_id", row.get("label_asym_id"))),
                "resSeq": int(row.get("auth_seq_id", row.get("label_seq_id", 0))),
                "iCode": pdb_icode,
                "x": float(row.get("Cartn_x", 0.0)),
                "y": float(row.get("Cartn_y", 0.0)),
                "z": float(row.get("Cartn_z", 0.0)),
                "occupancy": float(row.get("occupancy", 1.0)),
                "tempFactor": float(row.get("B_iso_or_equiv", 0.0)),
                "element": pdb_element,
                "charge": pdb_charge,
                "model": int(row.get("pdbx_PDB_model_num", 1)),
            }
        else:
            raise ValueError(f"Unsupported DataFrame format: {format_type}")

        # --- MODEL/ENDMDL Records ---
        current_model_num = atom_data["model"]
        if current_model_num != last_model_num:
            if last_model_num is not None:
                buffer.write("ENDMDL\n")
            buffer.write(f"MODEL     {current_model_num:>4}\n")
            last_model_num = current_model_num
            # Reset chain/residue tracking for the new model
            last_chain_id = None
            last_res_info = None

        # --- TER Records ---
        current_chain_id = atom_data["chainID"]
        current_res_info = (
            atom_data["resSeq"],
            atom_data["iCode"],
            atom_data["resName"],
        )

        # Write TER if chain ID changes within the same model
        if last_chain_id is not None and current_chain_id != last_chain_id:
            ter_serial = str(last_serial + 1).rjust(5)
            ter_res_name = last_res_info[2].strip().rjust(3)  # Use last residue's name
            ter_chain_id = last_chain_id
            ter_res_seq = str(last_res_info[0]).rjust(4)  # Use last residue's seq num
            ter_icode = (
                last_res_info[1] if last_res_info[1] else ""
            )  # Use last residue's icode

            ter_line = f"TER   {ter_serial}      {ter_res_name} {ter_chain_id}{ter_res_seq}{ter_icode}"
            buffer.write(ter_line.ljust(80) + "\n")

        # --- Format and Write ATOM/HETATM Line ---
        pdb_line = _format_pdb_atom_line(atom_data)
        buffer.write(pdb_line + "\n")

        # --- Update Tracking Variables ---
        last_serial = atom_data["serial"]
        last_chain_id = current_chain_id
        last_res_info = current_res_info

    # --- Final Records ---
    # Add TER record for the very last chain in the last model
    if last_chain_id is not None:
        ter_serial = str(last_serial + 1).rjust(5)
        ter_res_name = last_res_info[2].strip().rjust(3)
        ter_chain_id = last_chain_id
        ter_res_seq = str(last_res_info[0]).rjust(4)
        ter_icode = last_res_info[1] if last_res_info[1] else ""

        ter_line = f"TER   {ter_serial}      {ter_res_name} {ter_chain_id}{ter_res_seq}{ter_icode}"
        buffer.write(ter_line.ljust(80) + "\n")

    # Add ENDMDL if models were used
    if last_model_num is not None:
        buffer.write("ENDMDL\n")

    buffer.write("END\n")

    # --- Output Handling ---
    content = buffer.getvalue()
    buffer.close()

    if output is not None:
        if isinstance(output, str):
            with open(output, "w") as f:
                f.write(content)
        else:
            output.write(content)
        return None
    else:
        return content


def write_cif(
    df: pd.DataFrame, output: Union[str, TextIO, None] = None
) -> Union[str, None]:
    """
    Write a DataFrame of atom records to mmCIF format.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing atom records, as created by parse_pdb_atoms or parse_cif_atoms
    output : Union[str, TextIO, None], optional
        Output file path or file-like object. If None, returns the mmCIF content as a string.

    Returns:
    --------
    Union[str, None]
        If output is None, returns the mmCIF content as a string. Otherwise, returns None.
    """
    # Get the format of the DataFrame
    format_type = df.attrs.get("format", "PDB")

    # Create a new DataContainer
    data_container = DataContainer("rnapolis")

    # Define the attributes for atom_site category
    if format_type == "mmCIF":
        # Use existing mmCIF attributes
        attributes = list(df.columns)
    else:  # PDB format
        # Map PDB columns to mmCIF attributes
        attributes = [
            "group_PDB",  # record_type
            "id",  # serial
            "type_symbol",  # element
            "label_atom_id",  # name
            "label_alt_id",  # altLoc
            "label_comp_id",  # resName
            "label_asym_id",  # chainID
            "label_entity_id",  # (generated)
            "label_seq_id",  # resSeq
            "pdbx_PDB_ins_code",  # iCode
            "Cartn_x",  # x
            "Cartn_y",  # y
            "Cartn_z",  # z
            "occupancy",  # occupancy
            "B_iso_or_equiv",  # tempFactor
            "pdbx_formal_charge",  # charge
            "auth_seq_id",  # resSeq
            "auth_comp_id",  # resName
            "auth_asym_id",  # chainID
            "auth_atom_id",  # name
            "pdbx_PDB_model_num",  # model
        ]

    # Prepare rows for the atom_site category
    rows = []

    for _, row in df.iterrows():
        if format_type == "mmCIF":
            # Use existing mmCIF data, converting None to '?' universally
            row_data = []
            for attr in attributes:
                value = row.get(attr)
                if pd.isna(value):
                    # Use '?' as the standard placeholder for missing values
                    row_data.append("?")
                else:
                    # Ensure all non-missing values are converted to string
                    row_data.append(str(value))
        else:  # PDB format
            # Map PDB data to mmCIF format, converting None to '.' or '?'
            entity_id = "1"  # Default entity ID
            model_num = str(int(row["model"]))

            # Pre-process optional fields for mmCIF placeholders
            element_val = "?" if pd.isna(row.get("element")) else str(row["element"])
            altloc_val = "." if pd.isna(row.get("altLoc")) else str(row["altLoc"])
            icode_val = "." if pd.isna(row.get("iCode")) else str(row["iCode"])
            charge_val = "." if pd.isna(row.get("charge")) else str(row["charge"])

            row_data = [
                str(row["record_type"]),  # group_PDB
                str(int(row["serial"])),  # id
                element_val,  # type_symbol
                str(row["name"]),  # label_atom_id
                altloc_val,  # label_alt_id
                str(row["resName"]),  # label_comp_id
                str(row["chainID"]),  # label_asym_id
                entity_id,  # label_entity_id
                str(int(row["resSeq"])),  # label_seq_id
                icode_val,  # pdbx_PDB_ins_code
                f"{float(row['x']):.3f}",  # Cartn_x
                f"{float(row['y']):.3f}",  # Cartn_y
                f"{float(row['z']):.3f}",  # Cartn_z
                f"{float(row['occupancy']):.2f}",  # occupancy
                f"{float(row['tempFactor']):.2f}",  # B_iso_or_equiv
                charge_val,  # pdbx_formal_charge
                str(int(row["resSeq"])),  # auth_seq_id
                str(row["resName"]),  # auth_comp_id
                str(row["chainID"]),  # auth_asym_id
                str(row["name"]),  # auth_atom_id
                model_num,  # pdbx_PDB_model_num
            ]

        rows.append(row_data)

    # Create the atom_site category
    atom_site_category = DataCategory("atom_site", attributes, rows)

    # Add the category to the data container
    data_container.append(atom_site_category)

    # Create an IoAdapter for writing
    adapter = IoAdapterPy()

    # Handle output
    if output is None:
        # Return as string - write to a temporary file and read it back
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".cif") as temp_file:
            adapter.writeFile(temp_file.name, [data_container])
            temp_file.flush()
            temp_file.seek(0)
            return temp_file.read()
    elif isinstance(output, str):
        # Write to a file path
        adapter.writeFile(output, [data_container])
        return None
    else:
        # Write to a file-like object
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".cif") as temp_file:
            adapter.writeFile(temp_file.name, [data_container])
            temp_file.flush()
            temp_file.seek(0)
            output.write(temp_file.read())
        return None
