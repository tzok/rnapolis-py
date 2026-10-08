import logging
from typing import List
from enum import Enum
import pandas as pd

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    LeontisWesthof,
    Residue,
    ResidueAuth,
    ResidueLabel,
)
from rnapolis.tertiary import Structure3D


class InteractionType(Enum):
    BASE_PAIR = "base-pair"
    OTHER = "other"


def _has_alt_or_symmetry(row: pd.Series) -> bool:
    """Return True if any of the alternative location or symmetry fields are non-empty."""
    fields = [
        f"{field}{i}"
        for i in (1, 2)
        for field in ("alt", "label_alt_id", "symmetry_operation")
    ]
    for field in fields:
        value = row.get(field)
        if pd.isna(value):
            continue
        if str(value).strip() not in ("", "?", "."):
            return True
    return False


def _classify_interaction(row: pd.Series):
    """
    Classifies interaction based on DNATCO metrics.

    Returns:
        base-pair, LeontisWesthof
    """
    family = row.get("family", "")
    try:
        # Convert to the format expected by LeontisWesthof
        edge_type = family[0].lower()  # c or t
        edge1 = family[1].upper()  # W, H, S (convert to uppercase)
        edge2 = family[2].upper()  # W, H, S (convert to uppercase)

        lw_format = f"{edge_type}{edge1}{edge2}"
        return InteractionType.BASE_PAIR, LeontisWesthof[lw_format]
    except KeyError:
        logging.warning(f"DNATCO unknown interaction from family: {family}")
    return InteractionType.OTHER, None


def _parse_residues(row: pd.Series) -> tuple[Residue, Residue]:
    """
    Parse DNATCO row into 2 Residue objects.
    """
    residues = []

    def value(field: str):
        result = row.get(field)
        if pd.isna(result) or (
            isinstance(result, str) and result.strip() in ("", "?", ".")
        ):
            return None
        return result

    for i in range(1, 3):
        if f"auth_seq_id{i}" in row.index:
            label_chain = value(f"label_asym_id{i}")
            label_number = value(f"label_seq_id{i}")
            label_name = value(f"label_comp_id{i}")
            auth_chain = value(f"auth_asym_id{i}")
            auth_number = value(f"auth_seq_id{i}")
            auth_name = value(f"auth_comp_id{i}")
            insertion_code = value(f"pdbx_PDB_ins_code{i}")
        else:
            label_chain = label_number = label_name = None
            auth_chain = value(f"chain{i}")
            auth_number = value(f"nr{i}")
            auth_name = value(f"res{i}")
            insertion_code = value(f"ins{i}")

        label = None
        if label_number is not None and label_name is not None:
            label = ResidueLabel(label_chain, int(label_number), str(label_name))

        auth = None
        if auth_number is not None and auth_name is not None:
            auth = ResidueAuth(
                auth_chain,
                int(auth_number),
                insertion_code,
                str(auth_name),
            )
        residues.append(Residue(label, auth))

    return residues[0], residues[1]


def parse_dnatco_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse DNATCO output files and convert to BaseInteractions.

    Args:
        file_paths: List of paths to DNATCO output files containing basepair interactions

    Returns:
        BaseInteractions object containing the interactions found by DNATCO
    """

    bp_interactions: list[BasePair] = []

    # Process each input file
    for file_path in file_paths:
        logging.info(f"Processing DNATCO file: {file_path}")
        df = pd.read_csv(file_path)  # pyright: ignore[reportUnknownMemberType]
        for index, row in df.iterrows():
            assert isinstance(index, int)
            line_number = index + 2
            if _has_alt_or_symmetry(row):
                logging.warning(
                    f"Non-empty alt or symmetry_operation in {file_path} line: {line_number}"
                )
                continue

            r1, r2 = _parse_residues(row)
            interaction_type, interaction_subtype = _classify_interaction(row)
            if interaction_type == InteractionType.OTHER:
                continue
            assert interaction_subtype is not None
            interaction = BasePair(r1, r2, interaction_subtype, None)
            bp_interactions.append(interaction)

    return BaseInteractions.from_structure3d(  # pyright: ignore[reportUnknownMemberType]
        structure3d,
        bp_interactions,
        [],
        [],
        [],
        [],
    )
