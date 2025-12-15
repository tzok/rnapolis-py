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
)
from rnapolis.tertiary import Structure3D


class InteractionType(Enum):
    BASE_PAIR = "base-pair"
    OTHER = "other"


def _has_alt_or_symmetry(row: pd.Series) -> bool:
    """Return True if any of the alternative location or symmetry fields are non-empty."""
    fields = ["alt1", "symmetry_operation1", "alt2", "symmetry_operation2"]
    return any(
        str(row.get(field, "")).strip() not in ("", "None", "nan") for field in fields
    )


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


def _parse_residues(row: pd.Series):
    """
    Parse DNATCO row into 2 Residue objects.
    """
    res: List[ResidueAuth] = []
    for i in range(1, 3):
        chain = row[f"chain{i}"]
        number = int(row[f"nr{i}"])
        res_name = row[f"res{i}"]
        i_code = row.get(f"ins{i}", "").strip() or None
        res.append(ResidueAuth(chain, number, i_code, res_name))

    return Residue(None, res[0]), Residue(None, res[1])


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
