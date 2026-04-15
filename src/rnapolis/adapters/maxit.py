import logging
from tempfile import NamedTemporaryFile
from typing import List, Optional

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Saenger,
)
from rnapolis.metareader import read_metadata
from rnapolis.tertiary import Structure3D


def _maxit_convert_saenger(hbond_type_28: Optional[str]) -> Optional[Saenger]:
    if hbond_type_28 is None:
        return None
    try:
        index = int(hbond_type_28)
        if 1 <= index <= 28:
            return list(Saenger)[index - 1]
    except ValueError:
        pass
    return None


def _maxit_convert_lw(hbond_type_12: Optional[str]) -> Optional[LeontisWesthof]:
    if hbond_type_12 is None:
        return None
    try:
        index = int(hbond_type_12)
        if index == 1:
            return LeontisWesthof.cWW
        if index == 2:
            return LeontisWesthof.tWW
        if index == 3:
            return LeontisWesthof.cWH
        if index == 4:
            return LeontisWesthof.tWH
        if index == 5:
            return LeontisWesthof.cWS
        if index == 6:
            return LeontisWesthof.tWS
        if index == 7:
            return LeontisWesthof.cHH
        if index == 8:
            return LeontisWesthof.tHH
        if index == 9:
            return LeontisWesthof.cHS
        if index == 10:
            return LeontisWesthof.tHS
        if index == 11:
            return LeontisWesthof.cSS
        if index == 12:
            return LeontisWesthof.tSS
    except ValueError:
        pass
    return None


def parse_maxit_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse MAXIT output files and convert to BaseInteractions.

    MAXIT analysis is embedded in mmCIF files as ndb_struct_na_base_pair category.

    Args:
        file_paths: List of paths to mmCIF files containing MAXIT analysis

    Returns:
        BaseInteractions object containing the interactions found by MAXIT
    """
    all_base_pairs = []
    all_other_interactions = []

    # Find the first .cif file in the list
    cif_file = None
    for file_path in file_paths:
        if file_path.endswith(".cif"):
            cif_file = file_path
            break

    if cif_file is None:
        logging.warning("No .cif file found in MAXIT file list")
        return BaseInteractions([], [], [], [], [])

    # Log unused files
    unused_files = [f for f in file_paths if f != cif_file]
    if unused_files:
        logging.info(f"MAXIT: Using {cif_file}, ignoring unused files: {unused_files}")

    # Process only the first .cif file
    logging.info(f"Processing MAXIT file: {cif_file}")

    try:
        with open(cif_file, "r") as f:
            file_content = f.read()

        with NamedTemporaryFile("w+", suffix=".cif") as mmcif:
            mmcif.write(file_content)
            mmcif.seek(0)
            metadata = read_metadata(mmcif, ["ndb_struct_na_base_pair"])

        # Parse base pairs from this file
        for entry in metadata.get("ndb_struct_na_base_pair", []):
            auth_chain_i = entry.get("i_auth_asym_id")
            auth_seq_i = entry.get("i_auth_seq_id")
            auth_icode_i = entry.get("i_PDB_ins_code")
            name_i = entry.get("i_label_comp_id")

            auth_chain_j = entry.get("j_auth_asym_id")
            auth_seq_j = entry.get("j_auth_seq_id")
            auth_icode_j = entry.get("j_PDB_ins_code")
            name_j = entry.get("j_label_comp_id")

            label_chain_i = entry.get("i_label_asym_id")
            label_seq_i = entry.get("i_label_seq_id")

            label_chain_j = entry.get("j_label_asym_id")
            label_seq_j = entry.get("j_label_seq_id")

            # auth residue numbers are required
            if auth_seq_i is None or auth_seq_j is None:
                logging.warning(f"Skipping base pair with missing auth_seq_id: {entry}")
                continue

            auth_number_i = int(auth_seq_i)
            auth_number_j = int(auth_seq_j)

            auth_i = ResidueAuth(auth_chain_i, auth_number_i, auth_icode_i, name_i)
            auth_j = ResidueAuth(auth_chain_j, auth_number_j, auth_icode_j, name_j)

            label_i = None
            if (
                label_chain_i is not None
                and label_seq_i is not None
                and name_i is not None
            ):
                label_i = ResidueLabel(label_chain_i, int(label_seq_i), name_i)

            label_j = None
            if (
                label_chain_j is not None
                and label_seq_j is not None
                and name_j is not None
            ):
                label_j = ResidueLabel(label_chain_j, int(label_seq_j), name_j)

            residue_i = Residue(label_i, auth_i)
            residue_j = Residue(label_j, auth_j)

            saenger = _maxit_convert_saenger(entry.get("hbond_type_28"))
            lw = _maxit_convert_lw(entry.get("hbond_type_12"))

            if lw is not None:
                all_base_pairs.append(BasePair(residue_i, residue_j, lw, saenger))
            else:
                all_other_interactions.append(OtherInteraction(residue_i, residue_j))

    except Exception as e:
        logging.warning(f"Error processing MAXIT file {cif_file}: {e}", exc_info=True)

    return BaseInteractions.from_structure3d(
        structure3d, all_base_pairs, [], [], [], all_other_interactions
    )
