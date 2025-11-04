import logging
from typing import List, Optional

import orjson

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    LeontisWesthof,
    Residue,
    Stacking,
)
from rnapolis.tertiary import Structure3D


def _dssr_match_name_to_residue(
    structure3d: Structure3D, nt_id: Optional[str]
) -> Optional[Residue]:
    if nt_id is not None:
        nt_id = nt_id.split(":")[-1]
        for residue in structure3d.residues:
            if residue.full_name == nt_id:
                return Residue(residue.label, residue.auth)
        logging.warning(f"Failed to find residue {nt_id}")
    return None


def _dssr_match_lw(lw: Optional[str]) -> Optional[LeontisWesthof]:
    return LeontisWesthof[lw] if lw in dir(LeontisWesthof) else None


def parse_dssr_output(
    file_paths: List[str], structure3d: Structure3D, model: Optional[int] = None
) -> BaseInteractions:
    """
    Parse DSSR JSON output and convert to BaseInteractions.

    Args:
        file_paths: List of paths to DSSR output files
        structure3d: The 3D structure parsed from PDB/mmCIF
        model: Model number to use (if None, use first model)

    Returns:
        BaseInteractions object containing the interactions found by DSSR
    """
    base_pairs: List[BasePair] = []
    stackings: List[Stacking] = []

    # Find the first .json file in the list
    json_file = None
    for file_path in file_paths:
        if file_path.endswith(".json"):
            json_file = file_path
            break

    if json_file is None:
        logging.warning("No .json file found in DSSR file list")
        return BaseInteractions([], [], [], [], [])

    # Log unused files
    unused_files = [f for f in file_paths if f != json_file]
    if unused_files:
        logging.info(f"DSSR: Using {json_file}, ignoring unused files: {unused_files}")

    with open(json_file) as f:
        dssr = orjson.loads(f.read())

    # Handle multi-model files
    if "models" in dssr:
        if model is None and dssr.get("models"):
            # If model is None, use the first model
            dssr = dssr.get("models")[0].get("parameters", {})
        else:
            # Otherwise find the specified model
            for result in dssr.get("models", []):
                if result.get("model", None) == model:
                    dssr = result.get("parameters", {})
                    break

    for pair in dssr.get("pairs", []):
        nt1 = _dssr_match_name_to_residue(structure3d, pair.get("nt1", None))
        nt2 = _dssr_match_name_to_residue(structure3d, pair.get("nt2", None))
        lw = _dssr_match_lw(pair.get("LW", None))

        if nt1 is not None and nt2 is not None and lw is not None:
            base_pairs.append(BasePair(nt1, nt2, lw, None))

    for stack in dssr.get("stacks", []):
        nts = [
            _dssr_match_name_to_residue(structure3d, nt)
            for nt in stack.get("nts_long", "").split(",")
        ]
        for i in range(1, len(nts)):
            nt1 = nts[i - 1]
            nt2 = nts[i]
            if nt1 is not None and nt2 is not None:
                stackings.append(Stacking(nt1, nt2, None))

    return BaseInteractions.from_structure3d(
        structure3d, base_pairs, stackings, [], [], []
    )
