import io
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


def write_pdb(
    df: pd.DataFrame, output: Union[str, TextIO, None] = None
) -> Union[str, None]:
    """
    Write a DataFrame of atom records to PDB format.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing atom records, as created by parse_pdb_atoms or parse_cif_atoms
    output : Union[str, TextIO, None], optional
        Output file path or file-like object. If None, returns the PDB content as a string.

    Returns:
    --------
    Union[str, None]
        If output is None, returns the PDB content as a string. Otherwise, returns None.
    """
    # Create a buffer to store the PDB content
    buffer = io.StringIO()

    # Get the format of the DataFrame
    format_type = df.attrs.get("format", "PDB")

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        # Initialize the line with spaces
        line = " " * 80

        # Set record type (ATOM or HETATM)
        if format_type == "PDB":
            record_type = row["record_type"]
        else:  # mmCIF
            record_type = row.get("group_PDB", "ATOM")
        line = record_type.ljust(6) + line[6:]

        # Set atom serial number
        if format_type == "PDB":
            serial = str(int(row["serial"]))
        else:  # mmCIF
            serial = str(int(row["id"]))
        line = line[:6] + serial.rjust(5) + line[11:]

        # Set atom name
        if format_type == "PDB":
            atom_name = row["name"]
        else:  # mmCIF
            atom_name = row.get("auth_atom_id", row.get("label_atom_id", ""))

        # Right-justify atom name if it starts with a number
        if atom_name and atom_name[0].isdigit():
            line = line[:12] + atom_name.ljust(4) + line[16:]
        else:
            line = line[:12] + " " + atom_name.ljust(3) + line[16:]

        # Set alternate location indicator
        if format_type == "PDB":
            alt_loc = row.get("altLoc", "")
        else:  # mmCIF
            alt_loc = row.get("label_alt_id", "")
        line = line[:16] + alt_loc + line[17:]

        # Set residue name
        if format_type == "PDB":
            res_name = row["resName"]
        else:  # mmCIF
            res_name = row.get("auth_comp_id", row.get("label_comp_id", ""))
        line = line[:17] + res_name.ljust(3) + line[20:]

        # Set chain identifier
        if format_type == "PDB":
            chain_id = row["chainID"]
        else:  # mmCIF
            chain_id = row.get("auth_asym_id", row.get("label_asym_id", ""))
        line = line[:21] + chain_id + line[22:]

        # Set residue sequence number
        if format_type == "PDB":
            res_seq = str(int(row["resSeq"]))
        else:  # mmCIF
            res_seq = str(int(row.get("auth_seq_id", row.get("label_seq_id", 0))))
        line = line[:22] + res_seq.rjust(4) + line[26:]

        # Set insertion code
        if format_type == "PDB":
            icode = row["iCode"] if pd.notna(row["iCode"]) else ""
        else:  # mmCIF
            icode = (
                row.get("pdbx_PDB_ins_code", "")
                if pd.notna(row.get("pdbx_PDB_ins_code", ""))
                else ""
            )
        line = line[:26] + icode + line[27:]

        # Set X coordinate
        if format_type == "PDB":
            x = float(row["x"])
        else:  # mmCIF
            x = float(row["Cartn_x"])
        line = line[:30] + f"{x:8.3f}" + line[38:]

        # Set Y coordinate
        if format_type == "PDB":
            y = float(row["y"])
        else:  # mmCIF
            y = float(row["Cartn_y"])
        line = line[:38] + f"{y:8.3f}" + line[46:]

        # Set Z coordinate
        if format_type == "PDB":
            z = float(row["z"])
        else:  # mmCIF
            z = float(row["Cartn_z"])
        line = line[:46] + f"{z:8.3f}" + line[54:]

        # Set occupancy
        if format_type == "PDB":
            occupancy = float(row["occupancy"])
        else:  # mmCIF
            occupancy = float(row.get("occupancy", 1.0))
        line = line[:54] + f"{occupancy:6.2f}" + line[60:]

        # Set temperature factor
        if format_type == "PDB":
            temp_factor = float(row["tempFactor"])
        else:  # mmCIF
            temp_factor = float(row.get("B_iso_or_equiv", 0.0))
        line = line[:60] + f"{temp_factor:6.2f}" + line[66:]

        # Set element symbol
        if format_type == "PDB":
            element = row["element"]
        else:  # mmCIF
            element = row.get("type_symbol", "")
        line = line[:76] + element.rjust(2) + line[78:]

        # Set charge
        if format_type == "PDB":
            charge = row["charge"]
        else:  # mmCIF
            charge = row.get("pdbx_formal_charge", "")
            if charge and charge not in ["?", "."]:
                # Convert numeric charge to PDB format (e.g., "1+" or "2-")
                try:
                    charge_val = int(charge)
                    if charge_val != 0:
                        charge = f"{abs(charge_val)}{'+' if charge_val > 0 else '-'}"
                    else:
                        charge = ""
                except ValueError:
                    pass
        line = line[:78] + charge + line[80:]

        # Write the line to the buffer
        buffer.write(line.rstrip() + "\n")

    # Add END record
    buffer.write("END\n")

    # Get the content as a string
    content = buffer.getvalue()
    buffer.close()

    # Write to output if provided
    if output is not None:
        if isinstance(output, str):
            with open(output, "w") as f:
                f.write(content)
        else:
            output.write(content)
        return None

    # Return the content as a string
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
    data_container = DataContainer("data_structure")

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
            "pdbx_PDB_model_num",  # (generated)
        ]

    # Prepare rows for the atom_site category
    rows = []

    for _, row in df.iterrows():
        if format_type == "mmCIF":
            # Use existing mmCIF data
            row_data = [str(row.get(attr, "?")) for attr in attributes]
        else:  # PDB format
            # Map PDB data to mmCIF format
            entity_id = "1"  # Default entity ID
            model_num = "1"  # Default model number

            row_data = [
                str(row["record_type"]),  # group_PDB
                str(int(row["serial"])),  # id
                str(row["element"]),  # type_symbol
                str(row["name"]),  # label_atom_id
                str(row.get("altLoc", "")),  # label_alt_id
                str(row["resName"]),  # label_comp_id
                str(row["chainID"]),  # label_asym_id
                entity_id,  # label_entity_id
                str(int(row["resSeq"])),  # label_seq_id
                str(row["iCode"])
                if pd.notna(row["iCode"])
                else "?",  # pdbx_PDB_ins_code
                f"{float(row['x']):.3f}",  # Cartn_x
                f"{float(row['y']):.3f}",  # Cartn_y
                f"{float(row['z']):.3f}",  # Cartn_z
                f"{float(row['occupancy']):.2f}",  # occupancy
                f"{float(row['tempFactor']):.2f}",  # B_iso_or_equiv
                str(row.get("charge", "")) or "?",  # pdbx_formal_charge
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
