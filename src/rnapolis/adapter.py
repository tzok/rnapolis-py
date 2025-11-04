#! /usr/bin/env python
import argparse
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import orjson
from scipy.optimize import linear_sum_assignment

from rnapolis.annotator import (
    add_common_output_arguments,
    handle_output_arguments,
)
from rnapolis.common import (
    BR,
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BPh,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Saenger,
    Stacking,
    StackingTopology,
    Structure2D,
)
from rnapolis.metareader import read_metadata
from rnapolis.parser import read_3d_structure
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary import (
    Mapping2D3D,
    Structure3D,  # Import the new helper function
)
from rnapolis.tertiary_v2 import Structure as StructureV2
from rnapolis.util import handle_input_file


class ExternalTool(Enum):
    FR3D = "fr3d"
    DSSR = "dssr"
    RNAVIEW = "rnaview"
    BPNET = "bpnet"
    MAXIT = "maxit"
    BARNABA = "barnaba"
    MCANNOTATE = "mc-annotate"


logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


def auto_detect_tool(external_files: List[str]) -> ExternalTool:
    """
    Auto-detect the external tool based on file patterns.

    Args:
        external_files: List of external tool output file paths

    Returns:
        ExternalTool enum value based on detected patterns
    """
    if not external_files:
        return ExternalTool.MAXIT

    for file_path in external_files:
        basename = os.path.basename(file_path)

        # Check for FR3D pattern
        if basename.endswith("basepair_detail.txt"):
            return ExternalTool.FR3D

        # Check for RNAView pattern
        if basename.endswith(".out"):
            return ExternalTool.RNAVIEW

        # Check for BPNet pattern
        if basename.endswith("basepair.json"):
            return ExternalTool.BPNET

        # Check for MC-Annotate pattern
        if basename.endswith("stdout.txt"):
            return ExternalTool.MCANNOTATE

        # Check for Barnaba pattern
        if "pairing" in basename or "stacking" in basename:
            return ExternalTool.BARNABA

        # Check for JSON files (DSSR)
        if basename.endswith(".json"):
            return ExternalTool.DSSR

    # Default to MAXIT if no patterns match
    return ExternalTool.MAXIT


def parse_unit_id(nt: str) -> Residue:
    """Parse FR3D unit ID format into a Residue object."""
    fields = nt.split("|")
    icode = fields[7] if len(fields) >= 8 and fields[7] != "" else None
    auth = ResidueAuth(fields[2], int(fields[4]), icode, fields[3])
    return Residue(None, auth)


def unify_classification(fr3d_name: str) -> tuple:
    """Convert FR3D classification to internal format."""
    original_name = fr3d_name  # Keep for logging

    # Handle 'n' prefix (e.g., ncWW -> cWW, ns55 -> s55)
    if fr3d_name.startswith("n"):
        fr3d_name = fr3d_name[1:]
        logging.debug(
            f"Detected 'n' prefix: removed from {original_name} -> {fr3d_name}"
        )

    # Handle alternative base pairs with 'a' suffix (e.g., cWWa -> cWW)
    if len(fr3d_name) >= 3 and fr3d_name.endswith("a"):
        fr3d_name = fr3d_name[:-1]  # Remove the 'a' suffix
        logging.debug(
            f"Detected alternative base pair: removed 'a' suffix from {original_name} -> {fr3d_name}"
        )

    # Handle backbone interactions: 0BR, 1BR, ... 9BR for base-ribose
    if len(fr3d_name) == 3 and fr3d_name[1:] == "BR" and fr3d_name[0].isdigit():
        try:
            br_type = f"_{fr3d_name[0]}"
            return ("base-ribose", BR[br_type])
        except (ValueError, KeyError):
            logging.debug(f"Unknown base-ribose interaction: {original_name}")
            return ("other", None)

    # Handle backbone interactions: 0BPh, 1BPh, ... 9BPh for base-phosphate
    if len(fr3d_name) == 4 and fr3d_name[1:] == "BPh" and fr3d_name[0].isdigit():
        try:
            bph_type = f"_{fr3d_name[0]}"
            return ("base-phosphate", BPh[bph_type])
        except (ValueError, KeyError):
            logging.debug(f"Unknown base-phosphate interaction: {original_name}")
            return ("other", None)

    # Handle the stacking notation from direct FR3D service (s33, s35, s53, s55)
    if (
        len(fr3d_name) == 3
        and fr3d_name.startswith("s")
        and fr3d_name[1] in ("3", "5")
        and fr3d_name[2] in ("3", "5")
    ):
        if fr3d_name == "s33":
            return ("stacking", StackingTopology.downward)
        if fr3d_name == "s55":
            return ("stacking", StackingTopology.upward)
        if fr3d_name == "s35":
            return ("stacking", StackingTopology.outward)
        if fr3d_name == "s53":
            return ("stacking", StackingTopology.inward)

    # Handle the cWW style notation from direct FR3D service output
    # Support both uppercase and lowercase edge names (e.g., cWW, cww, tHS, ths, tSs, etc.)
    if len(fr3d_name) == 3 and fr3d_name[0].lower() in ("c", "t"):
        try:
            # Convert to the format expected by LeontisWesthof
            edge_type = fr3d_name[0].lower()  # c or t
            edge1 = fr3d_name[1].upper()  # W, H, S (convert to uppercase)
            edge2 = fr3d_name[2].upper()  # W, H, S (convert to uppercase)

            lw_format = f"{edge_type}{edge1}{edge2}"
            return ("base-pair", LeontisWesthof[lw_format])
        except KeyError:
            logging.debug(
                f"Fr3d unknown interaction from service: {original_name} -> {fr3d_name}"
            )
            return ("other", None)

    # Handle other classifications with different formatting
    logging.debug(f"Fr3d unknown interaction: {fr3d_name}")
    return ("other", None)


def _process_interaction_line(
    line: str,
    interactions_data: Dict[str, list],
):
    """
    Process a single interaction line and add it to the appropriate list.

    Args:
        line: The tab-separated interaction line
        interactions_data: Dictionary containing all interaction lists

    Returns:
        True if successfully processed, False otherwise
    """
    try:
        # Split by tabs and get the first three fields
        parts = line.split("\t")
        if len(parts) < 3:
            logging.warning(f"Invalid interaction line format: {line}")
            return False

        nt1 = parts[0]
        interaction_type = parts[1]
        nt2 = parts[2]

        nt1_residue = parse_unit_id(nt1)
        nt2_residue = parse_unit_id(nt2)

        # Convert the interaction type to our internal format
        interaction_category, classification = unify_classification(interaction_type)

        # Add to the appropriate list based on the interaction category
        if interaction_category == "base-pair":
            interactions_data["base_pairs"].append(
                BasePair(nt1_residue, nt2_residue, classification, None)
            )
        elif interaction_category == "stacking":
            interactions_data["stackings"].append(
                Stacking(nt1_residue, nt2_residue, classification)
            )
        elif interaction_category == "base-ribose":
            interactions_data["base_ribose_interactions"].append(
                BaseRibose(nt1_residue, nt2_residue, classification)
            )
        elif interaction_category == "base-phosphate":
            interactions_data["base_phosphate_interactions"].append(
                BasePhosphate(nt1_residue, nt2_residue, classification)
            )
        elif interaction_category == "other":
            interactions_data["other_interactions"].append(
                OtherInteraction(nt1_residue, nt2_residue)
            )

        return True
    except (ValueError, IndexError) as e:
        logging.warning(f"Error parsing interaction: {e}")
        return False


def match_dssr_name_to_residue(
    structure3d: Structure3D, nt_id: Optional[str]
) -> Optional[Residue]:
    if nt_id is not None:
        nt_id = nt_id.split(":")[-1]
        for residue in structure3d.residues:
            if residue.full_name == nt_id:
                return Residue(residue.label, residue.auth)
        logging.warning(f"Failed to find residue {nt_id}")
    return None


def match_dssr_lw(lw: Optional[str]) -> Optional[LeontisWesthof]:
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
        nt1 = match_dssr_name_to_residue(structure3d, pair.get("nt1", None))
        nt2 = match_dssr_name_to_residue(structure3d, pair.get("nt2", None))
        lw = match_dssr_lw(pair.get("LW", None))

        if nt1 is not None and nt2 is not None and lw is not None:
            base_pairs.append(BasePair(nt1, nt2, lw, None))

    for stack in dssr.get("stacks", []):
        nts = [
            match_dssr_name_to_residue(structure3d, nt)
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


def _maxit_convert_saenger(hbond_type_28: str) -> Optional[Saenger]:
    if hbond_type_28 == "?":
        return None
    try:
        index = int(hbond_type_28)
        if 1 <= index <= 28:
            return list(Saenger)[index - 1]
    except ValueError:
        pass
    return None


def _maxit_convert_lw(hbond_type_12) -> Optional[LeontisWesthof]:
    if hbond_type_12 == "?":
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
            auth_chain_i = entry["i_auth_asym_id"]
            auth_number_i = int(entry["i_auth_seq_id"])
            auth_icode_i = (
                entry["i_PDB_ins_code"] if entry["i_PDB_ins_code"] != "?" else None
            )
            name_i = entry["i_label_comp_id"]
            auth_i = ResidueAuth(auth_chain_i, auth_number_i, auth_icode_i, name_i)

            auth_chain_j = entry["j_auth_asym_id"]
            auth_number_j = int(entry["j_auth_seq_id"])
            auth_icode_j = (
                entry["j_PDB_ins_code"] if entry["j_PDB_ins_code"] != "?" else None
            )
            name_j = entry["j_label_comp_id"]
            auth_j = ResidueAuth(auth_chain_j, auth_number_j, auth_icode_j, name_j)

            label_chain_i = entry["i_label_asym_id"]
            label_number_i = int(entry["i_label_seq_id"])
            label_i = ResidueLabel(label_chain_i, label_number_i, name_i)

            label_chain_j = entry["j_label_asym_id"]
            label_number_j = int(entry["j_label_seq_id"])
            label_j = ResidueLabel(label_chain_j, label_number_j, name_j)

            residue_i = Residue(label_i, auth_i)
            residue_j = Residue(label_j, auth_j)

            saenger = _maxit_convert_saenger(entry["hbond_type_28"])
            lw = _maxit_convert_lw(entry["hbond_type_12"])

            if lw is not None:
                all_base_pairs.append(BasePair(residue_i, residue_j, lw, saenger))
            else:
                all_other_interactions.append(OtherInteraction(residue_i, residue_j))

    except Exception as e:
        logging.warning(f"Error processing MAXIT file {cif_file}: {e}", exc_info=True)

    return BaseInteractions.from_structure3d(
        structure3d, all_base_pairs, [], [], [], all_other_interactions
    )


def _bpnet_convert_lw(bpnet_lw) -> LeontisWesthof:
    """Convert BPNet LW notation to LeontisWesthof enum."""
    if len(bpnet_lw) != 4:
        raise ValueError(f"bpnet lw invalid length: {bpnet_lw}")
    bpnet_lw = bpnet_lw.replace("+", "W").replace("z", "S").replace("g", "H")
    edge5 = bpnet_lw[0].upper()
    edge3 = bpnet_lw[2].upper()
    stericity = bpnet_lw[3].lower()
    return LeontisWesthof[f"{stericity}{edge5}{edge3}"]


def _bpnet_residues_from_overlap_info(fields):
    """Parse residue information from overlap line fields."""
    chains = fields[6].split("^")
    numbers = list(map(int, fields[3].split(":")))
    icode1, icode2 = fields[2], fields[4]
    names = fields[5].split(":")

    if icode1 in " ?":
        icode1 = None
    if icode2 in " ?":
        icode2 = None

    nt1 = Residue(None, ResidueAuth(chains[0], numbers[0], icode1, names[0]))
    nt2 = Residue(None, ResidueAuth(chains[1], numbers[1], icode2, names[1]))
    return nt1, nt2


def parse_bpnet_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse BPNet output files and convert to BaseInteractions.

    Args:
        file_paths: List of paths to BPNet output files (basepair.json and .rob files)

    Returns:
        BaseInteractions object containing the interactions found by BPNet
    """
    # Find required files
    basepair_json = None
    rob_file = None

    for file_path in file_paths:
        if file_path.endswith("basepair.json"):
            basepair_json = file_path
        elif file_path.endswith(".rob"):
            rob_file = file_path

    # Log unused files
    used_files = [f for f in [basepair_json, rob_file] if f is not None]
    unused_files = [f for f in file_paths if f not in used_files]
    if unused_files:
        logging.info(
            f"BPNet: Using {used_files}, ignoring unused files: {unused_files}"
        )

    base_pairs = []
    stackings = []
    base_ribose_interactions = []
    base_phosphate_interactions = []
    other_interactions = []

    # Parse base pairs from JSON file
    if basepair_json:
        logging.info(f"Processing BPNet basepair file: {basepair_json}")
        try:
            with open(basepair_json, encoding="utf-8") as f:
                data = orjson.loads(f.read())

            for entry in data["basepairs"]:
                nt1 = Residue(
                    None,
                    ResidueAuth(
                        entry["chain1"],
                        entry["resnum1"],
                        entry["ins1"],
                        entry["resname1"],
                    ),
                )
                nt2 = Residue(
                    None,
                    ResidueAuth(
                        entry["chain2"],
                        entry["resnum2"],
                        entry["ins2"],
                        entry["resname2"],
                    ),
                )
                lw = _bpnet_convert_lw(entry["basepair"])
                base_pairs.append(BasePair(nt1, nt2, lw, None))
        except Exception as e:
            logging.warning(
                f"Error processing BPNet basepair file {basepair_json}: {e}",
                exc_info=True,
            )

    # Parse overlaps from ROB file
    if rob_file:
        logging.info(f"Processing BPNet rob file: {rob_file}")
        try:
            with open(rob_file, encoding="utf-8") as f:
                rob_content = f.read()

            for line in rob_content.splitlines():
                if line.startswith("OVLP"):
                    fields = line.strip().split()
                    if len(fields) == 13:
                        # ASTK means Adjacent Stacking, OSTK means Non-Adjacent Stacking
                        # ADJA means Adjacent contact but not proper stacking
                        if fields[7] in ["ASTK", "OSTK", "ADJA"]:
                            nt1, nt2 = _bpnet_residues_from_overlap_info(fields)
                            stackings.append(Stacking(nt1, nt2, None))
                    else:
                        logging.warning(f"Failed to parse OVLP line: {line}")
                elif line.startswith("PROX"):
                    fields = line.strip().split()
                    if len(fields) == 11:
                        nt1, nt2 = _bpnet_residues_from_overlap_info(fields)
                        atom1, atom2 = fields[7].split(":")

                        # Determine element types based on atom names
                        phosphate_atoms = frozenset(
                            (
                                "P",
                                "OP1",
                                "OP2",
                                "O5'",
                                "C5'",
                                "C4'",
                                "C3'",
                                "O3'",
                                "O5*",
                                "C5*",
                                "C4*",
                                "C3*",
                                "O3*",
                            )
                        )
                        ribose_atoms = frozenset(
                            ("C1'", "C2'", "O2'", "O4'", "C1*", "C2*", "O2*", "O4*")
                        )
                        base_atoms = frozenset(
                            (
                                "C2",
                                "C4",
                                "C5",
                                "C6",
                                "C8",
                                "N1",
                                "N2",
                                "N3",
                                "N4",
                                "N6",
                                "N7",
                                "N9",
                                "O2",
                                "O4",
                                "O6",
                            )
                        )

                        def assign_element(atom_name):
                            if atom_name in phosphate_atoms:
                                return "PHOSPHATE"
                            elif atom_name in ribose_atoms:
                                return "RIBOSE"
                            elif atom_name in base_atoms:
                                return "BASE"
                            else:
                                return "UNKNOWN"

                        element1 = assign_element(atom1)
                        element2 = assign_element(atom2)

                        # Base-ribose interactions
                        if element1 == "BASE" and element2 == "RIBOSE":
                            base_ribose_interactions.append(BaseRibose(nt1, nt2, None))
                        elif element1 == "RIBOSE" and element2 == "BASE":
                            base_ribose_interactions.append(BaseRibose(nt2, nt1, None))

                        # Base-phosphate interactions
                        elif element1 == "BASE" and element2 == "PHOSPHATE":
                            base_phosphate_interactions.append(
                                BasePhosphate(nt1, nt2, None)
                            )
                        elif element1 == "PHOSPHATE" and element2 == "BASE":
                            base_phosphate_interactions.append(
                                BasePhosphate(nt2, nt1, None)
                            )

                        # Other interactions
                        other_interactions.append(OtherInteraction(nt1, nt2))
                    else:
                        logging.warning(f"Failed to parse PROX line: {line}")
        except Exception as e:
            logging.warning(
                f"Error processing BPNet rob file {rob_file}: {e}", exc_info=True
            )

    return BaseInteractions.from_structure3d(
        structure3d,
        base_pairs,
        stackings,
        base_ribose_interactions,
        base_phosphate_interactions,
        other_interactions,
    )


@dataclass
class _RNAViewPotentialResidue:
    residue: Residue
    position_c2: Optional[Tuple[float, float, float]]
    position_c6: Optional[Tuple[float, float, float]]
    position_n1: Optional[Tuple[float, float, float]]

    def is_correct_according_to_rnaview(self) -> bool:
        """
        This is a reimplementation of residue_ident() function from fpair_sub.c from RNAView source code.
        """
        if any(
            (
                self.position_c2 is None,
                self.position_c6 is None,
                self.position_n1 is None,
            )
        ):
            return False

        distance_n1_c2 = math.dist(self.position_n1, self.position_c2)  # type: ignore
        distance_n1_c6 = math.dist(self.position_n1, self.position_c6)  # type: ignore
        distance_c2_c6 = math.dist(self.position_c2, self.position_c6)  # type: ignore
        return all(
            (distance_n1_c2 <= 2.0, distance_n1_c6 <= 2.0, distance_c2_c6 <= 3.0)
        )


# RNAView regex pattern from the reference implementation
_RNAVIEW_REGEX = re.compile(
    r"\s*(\d+)_(\d+),\s+(\w):\s+(-?\d+)\s+(\w+)-(\w+)\s+(-?\d+)\s+(\w):\s+(syn|\s+)*((./.)\s+(cis|tran)(syn|\s+)*([IVX,]+|n/a|![^.]+)|stacked)\.?"
)

# RNAView tokens
_RNAVIEW_BEGIN_BASE_PAIR = "BEGIN_base-pair"
_RNAVIEW_END_BASE_PAIR = "END_base-pair"
_RNAVIEW_STACKING = "stacked"
_RNAVIEW_BASE_RIBOSE = "!(b_s)"
_RNAVIEW_BASE_PHOSPHATE = "!b_(O1P,O2P)"
_RNAVIEW_OTHER_INTERACTION = "!(s_s)"
_RNAVIEW_SAENGER_UNKNOWN = "n/a"
_RNAVIEW_PLUS_INTERACTION = "+/+"  # For us - cWW
_RNAVIEW_MINUS_INTERACTION = "-/-"  # For us - cWW
_RNAVIEW_X_INTERACTION = "X/X"  # For us - cWW
_RNAVIEW_ONE_HBOND = "!1H(b_b)"  # For us - OtherInteraction
_RNAVIEW_DOUBLE_SAENGER = ("XIV,XV", "XII,XIII")
_RNAVIEW_UNKNOWN_LW_CHARS = (".", "?")
_RNAVIEW_ROMAN_NUMERALS = ("I", "V", "X")


def _rnaview_get_leontis_westhof(
    lw_info: str, trans_cis_info: str
) -> Optional[LeontisWesthof]:
    """Convert RNAView LW notation to LeontisWesthof enum."""
    trans_cis = trans_cis_info[0]
    if any(char in lw_info for char in _RNAVIEW_UNKNOWN_LW_CHARS):
        return None
    if lw_info in (
        _RNAVIEW_PLUS_INTERACTION,
        _RNAVIEW_MINUS_INTERACTION,
        _RNAVIEW_X_INTERACTION,
    ):
        return LeontisWesthof[f"{trans_cis}WW"]
    return LeontisWesthof[f"{trans_cis}{lw_info[0].upper()}{lw_info[2].upper()}"]


def _rnaview_append_residues_from_input_using_rnaview_indexing(
    input_content: str, input_type: str = "cif"
) -> Dict[int, Residue]:
    """Parse input content and create RNAView-style residue mapping."""
    atoms_df = (
        parse_cif_atoms(input_content)
        if input_type == "cif"
        else parse_pdb_atoms(input_content)
    )
    structure = StructureV2(atoms_df)
    residues_from_pdb: Dict[int, Residue] = {}
    counter = 1

    for residue in structure.residues:
        residue_common = Residue(
            None,
            ResidueAuth(
                residue.chain_id,
                residue.residue_number,
                residue.insertion_code,
                residue.residue_name,
            ),
        )
        c2 = residue.find_atom("C2")
        c6 = residue.find_atom("C6")
        n1 = residue.find_atom("N1")
        potential_residue = _RNAViewPotentialResidue(
            residue_common,
            c2.coordinates if c2 else None,
            c6.coordinates if c6 else None,
            n1.coordinates if n1 else None,
        )
        if potential_residue.is_correct_according_to_rnaview():
            residues_from_pdb[counter] = potential_residue.residue
            counter += 1

    logging.debug("RNAView residues mapping:")
    for idx, residue in sorted(residues_from_pdb.items()):
        logging.debug(f"  {idx}: {residue}")

    return residues_from_pdb


def _rnaview_check_indexing_correctness(
    regex_result: Tuple[str, ...], line: str, residues_from_pdb: Dict[int, Residue]
) -> None:
    """Check if RNAView internal indexing matches PDB residue information."""
    residue_left = residues_from_pdb[int(regex_result[0])]

    if residue_left.auth.chain.lower() != regex_result[
        2
    ].lower() or residue_left.auth.number != int(regex_result[3]):
        raise ValueError(
            f"Wrong internal index for {residue_left}. Fix RNAView internal index mapping. Line: {line}"
        )

    residue_right = residues_from_pdb[int(regex_result[1])]

    if residue_right.auth.chain.lower() != regex_result[
        7
    ].lower() or residue_right.auth.number != int(regex_result[6]):
        raise ValueError(
            f"Wrong internal index for {residue_right}. Fix RNAView internal index mapping. Line: {line}"
        )


def parse_rnaview_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse RNAView output files and convert to BaseInteractions.

    Args:
        file_paths: List of paths to RNAView output files (.out files)
        structure3d: The 3D structure parsed from PDB/mmCIF

    Returns:
        BaseInteractions object containing the interactions found by RNAView
    """
    # Find the first .out file in the list
    out_file = None
    pdb_file = None
    cif_file = None
    for file_path in file_paths:
        if file_path.endswith(".out"):
            out_file = file_path
        elif file_path.endswith(".pdb"):
            pdb_file = file_path
        elif file_path.endswith(".cif"):
            cif_file = file_path

    if out_file is None:
        logging.warning("No .out file found in RNAView file list")
        return BaseInteractions([], [], [], [], [])

    # Log unused files
    used_files = [f for f in [out_file, pdb_file, cif_file] if f is not None]
    unused_files = [f for f in file_paths if f not in used_files]
    if unused_files:
        logging.info(
            f"RNAView: Using {used_files}, ignoring unused files: {unused_files}"
        )

    base_pairs = []
    stackings = []
    base_ribose_interactions = []
    base_phosphate_interactions = []
    other_interactions = []

    # Parse PDB content to build residue mapping if PDB file is available
    residues_from_input: Dict[int, Residue] = {}

    if cif_file:
        input_file = cif_file
        input_type = "cif"
    elif pdb_file:
        input_file = pdb_file
        input_type = "pdb"
    else:
        input_file = None
        input_type = None

    if input_file:
        logging.info(f"Processing RNAView mmCIF file: {input_file}")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_content = f.read()
            residues_from_input = _rnaview_append_residues_from_input_using_rnaview_indexing(
                input_content, input_type
            )
        except Exception as e:
            logging.warning(
                f"Error processing RNAView mmCIF file {cif_file}: {e}", exc_info=True
            )

    # Process the RNAView output file
    logging.info(f"Processing RNAView file: {out_file}")

    try:
        with open(out_file, "r", encoding="utf-8") as f:
            rnaview_result = f.read()

        base_pair_section = False
        for line in rnaview_result.splitlines():
            if line.startswith(_RNAVIEW_BEGIN_BASE_PAIR):
                base_pair_section = True
            elif line.startswith(_RNAVIEW_END_BASE_PAIR):
                base_pair_section = False
            elif base_pair_section:
                rnaview_regex_result = re.search(_RNAVIEW_REGEX, line)
                if rnaview_regex_result is None:
                    logging.warning(f"RNAView regex failed for line: {line}")
                    continue

                rnaview_regex_groups = rnaview_regex_result.groups()

                # Log parsed groups with their meanings
                logging.debug("RNAView regex parsed:")
                logging.debug(
                    f"  First residue:  idx={rnaview_regex_groups[0]}, chain={rnaview_regex_groups[2]}, num={rnaview_regex_groups[3]}, name={rnaview_regex_groups[4]}"
                )
                logging.debug(
                    f"  Second residue: idx={rnaview_regex_groups[1]}, chain={rnaview_regex_groups[7]}, num={rnaview_regex_groups[6]}, name={rnaview_regex_groups[5]}"
                )
                if rnaview_regex_groups[9] == "stacked":
                    logging.debug("  Interaction: stacking")
                else:
                    logging.debug(f"  LW edges: {rnaview_regex_groups[10]}")
                    logging.debug(f"  LW orientation: {rnaview_regex_groups[11]}")
                    logging.debug(f"  Classification: {rnaview_regex_groups[13]}")

                # Use residue mapping if available, otherwise create residues from regex
                if residues_from_input:
                    try:
                        # TODO: this check fails for two-letter chain names, because RNAView output uses only one letter anyway
                        # check_indexing_correctness(
                        #     rnaview_regex_groups, line, residues_from_input
                        # )
                        residue_left = residues_from_input[int(rnaview_regex_groups[0])]
                        residue_right = residues_from_input[
                            int(rnaview_regex_groups[1])
                        ]
                    except (KeyError, ValueError) as e:
                        logging.warning(f"RNAView indexing error: {e}")
                        continue
                else:
                    # Fallback: create residues from regex groups
                    chain_left = rnaview_regex_groups[2]
                    number_left = int(rnaview_regex_groups[3])
                    name_left = rnaview_regex_groups[4]

                    chain_right = rnaview_regex_groups[7]
                    number_right = int(rnaview_regex_groups[6])
                    name_right = rnaview_regex_groups[5]

                    residue_left = Residue(
                        None, ResidueAuth(chain_left, number_left, None, name_left)
                    )
                    residue_right = Residue(
                        None, ResidueAuth(chain_right, number_right, None, name_right)
                    )

                # Interaction OR Saenger OR n/a OR empty string
                token = rnaview_regex_groups[13]

                if rnaview_regex_groups[9] == _RNAVIEW_STACKING:
                    stackings.append(Stacking(residue_left, residue_right, None))

                elif token == _RNAVIEW_BASE_RIBOSE:
                    base_ribose_interactions.append(
                        BaseRibose(residue_left, residue_right, None)
                    )

                elif token == _RNAVIEW_BASE_PHOSPHATE:
                    base_phosphate_interactions.append(
                        BasePhosphate(residue_left, residue_right, None)
                    )

                elif token in (_RNAVIEW_OTHER_INTERACTION, _RNAVIEW_ONE_HBOND):
                    other_interactions.append(
                        OtherInteraction(residue_left, residue_right)
                    )

                elif token == _RNAVIEW_SAENGER_UNKNOWN:
                    leontis_westhof = _rnaview_get_leontis_westhof(
                        rnaview_regex_groups[10], rnaview_regex_groups[11]
                    )
                    if leontis_westhof is None:
                        other_interactions.append(
                            OtherInteraction(residue_left, residue_right)
                        )
                    else:
                        base_pairs.append(
                            BasePair(residue_left, residue_right, leontis_westhof, None)
                        )

                elif (
                    all(char in _RNAVIEW_ROMAN_NUMERALS for char in token)
                    or token in _RNAVIEW_DOUBLE_SAENGER
                ):
                    leontis_westhof = _rnaview_get_leontis_westhof(
                        rnaview_regex_groups[10], rnaview_regex_groups[11]
                    )
                    if leontis_westhof is None:
                        other_interactions.append(
                            OtherInteraction(residue_left, residue_right)
                        )
                    else:
                        saenger = (
                            Saenger[token.split(",", 1)[0]]
                            if token in _RNAVIEW_DOUBLE_SAENGER
                            else Saenger[token]
                        )
                        base_pairs.append(
                            BasePair(
                                residue_left, residue_right, leontis_westhof, saenger
                            )
                        )

                else:
                    logging.warning(f"Unknown RNAView interaction: {token}")

    except Exception as e:
        logging.warning(f"Error processing RNAView file {out_file}: {e}", exc_info=True)

    return BaseInteractions.from_structure3d(
        structure3d,
        base_pairs,
        stackings,
        base_ribose_interactions,
        base_phosphate_interactions,
        other_interactions,
    )


_BARNABA_STACKING_TOPOLOGIES = {
    ">>": "upward",
    "<<": "downward",
    "<>": "outward",
    "><": "inward",
}


def _barnaba_assign_indices_to_chains(indices, chains, score, avail):
    # Convert to numpy for Hungarian; fall back to greedy if SciPy not available
    S = np.array(score, dtype=float)
    A = np.array(avail, dtype=float)
    # small tie-breaker: prefer chains that have the residues at all
    S2 = S + 1e-6 * A
    # Maximize S2 by minimizing -S2
    row_ind, col_ind = linear_sum_assignment(-S2)

    mapping = {}
    pair_scores = {}
    for i, j in zip(row_ind, col_ind):
        mapping[indices[i]] = chains[j]
        pair_scores[(indices[i], chains[j])] = int(score[i][j])
    return mapping, pair_scores


def _barnaba_map_barnaba_to_rnapolis(
    barnaba: Iterable[Tuple[str, int, int]],
    rnapolis: Iterable[Tuple[str, int, str]],
    name_eq: Callable[[str, str], bool] = lambda a, b: a == b,
):
    # Index input
    barnaba = list(barnaba)
    rnapolis = list(rnapolis)

    # rnapolis grouped by number, and fast lookup by (number, chain)
    rnap_by_num = defaultdict(dict)  # number -> {chain: name}
    rnap_by_num_chain = {}  # (number, chain) -> (name, number, chain)
    chains_all = set()
    for rname, num, chain in rnapolis:
        rnap_by_num[num][chain] = rname
        rnap_by_num_chain[(num, chain)] = (rname, num, chain)
        chains_all.add(chain)

    indices = sorted({idx for _, _, idx in barnaba})
    chains = sorted(chains_all)
    idx_to_row = {idx: i for i, idx in enumerate(indices)}
    ch_to_col = {ch: j for j, ch in enumerate(chains)}

    # Build score and availability matrices
    # score[i, c] = number of name matches if index i -> chain c
    # avail[i, c] = number of residues where index i has a number that exists in chain c
    m, n = len(indices), len(chains)
    score = [[0] * n for _ in range(m)]
    avail = [[0] * n for _ in range(m)]
    for bname, num, idx in barnaba:
        row = idx_to_row[idx]
        for ch, rname in rnap_by_num.get(num, {}).items():
            col = ch_to_col[ch]
            avail[row][col] += 1
            if name_eq(bname, rname):
                score[row][col] += 1

    # Choose assignment that maximizes score (with tiny tie-break on availability)
    mapping, pair_scores = _barnaba_assign_indices_to_chains(
        indices, chains, score, avail
    )

    # Use mapping to pair each barnaba residue to its rnapolis counterpart
    pairs = []
    mismatches = []
    missing = []
    for bname, num, idx in barnaba:
        ch = mapping[idx]
        r = rnap_by_num_chain.get((num, ch))
        if r is None:
            missing.append((bname, num, idx))
            continue
        match = bool(name_eq(bname, r[0]))
        pairs.append(((bname, num, idx), r, match))
        if not match:
            mismatches.append(((bname, num, idx), r))

    # Optional: extras present only in rnapolis
    barnaba_keys = {(num, mapping[idx]) for _, num, idx in barnaba}
    extras_in_rnapolis = [r for r in rnapolis if (r[1], r[2]) not in barnaba_keys]

    return {
        "index_to_chain": mapping,  # dict: barnaba_index -> rnapolis_chain
        "pair_scores": pair_scores,  # per (index, chain) chosen, how many name matches
        "pairs": pairs,  # list of ((bname,num,idx), (rname,num,chain), name_match_bool)
        "mismatches": mismatches,  # list of pairs where names differ
        "missing_in_rnapolis": missing,  # barnaba residues lacking a counterpart (should be empty)
        "extras_in_rnapolis": extras_in_rnapolis,
    }


def _barnaba_get_leontis_westhof(interaction: str) -> Optional[LeontisWesthof]:
    if "x" in interaction.lower():
        return None
    if interaction in ("WCc", "GUc"):
        return LeontisWesthof.cWW
    return LeontisWesthof[f"{interaction[2]}{interaction[:2]}"]


def _barnaba_get_residue(
    residue_info: str,
    barnaba_mapping: Dict,
    residue_mapping: Dict,
    rnapolis_mapping: Dict,
) -> Optional[Residue]:
    barnaba_tuple = barnaba_mapping.get(residue_info, None)
    rnapolis_tuple = residue_mapping.get(barnaba_tuple, None)
    return rnapolis_mapping.get(rnapolis_tuple, None)


def parse_barnaba_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse barnaba output files and convert to BaseInteractions.
    Args:
        file_paths: List of paths to barnaba output files (pairing and stacking)
        structure3d: The 3D structure parsed from PDB/mmCIF
    Returns:
        BaseInteractions object containing the interactions found by barnaba
    """
    pairing_file = None
    stacking_file = None

    for file_path in file_paths:
        if "pairing" in os.path.basename(file_path):
            pairing_file = file_path
        elif "stacking" in os.path.basename(file_path):
            stacking_file = file_path

    if pairing_file is None and stacking_file is None:
        logging.warning("No barnaba pairing or stacking files found")
        return BaseInteractions([], [], [], [], [])

    barnaba_mapping: List[str] = []

    with open(pairing_file or stacking_file, "r") as f:
        for line in f.readlines():
            if line.startswith("# sequence"):
                barnaba_mapping = line.strip().split()[2].split("-")
                break

    if not barnaba_mapping:
        logging.warning("Could not find barnaba sequence in output files")
        return BaseInteractions([], [], [], [], [])

    barnaba_mapping = {
        residue_info: (
            residue_info.split("_")[0],
            int(residue_info.split("_")[1]),
            int(residue_info.split("_")[2]),
        )
        for residue_info in barnaba_mapping
    }
    rnapolis_mapping = {
        (residue.auth.name, residue.auth.number, residue.auth.chain): residue
        for residue in structure3d.residues
        if residue.auth and residue.is_nucleotide
    }
    barnaba_to_rnapolis_mapping = _barnaba_map_barnaba_to_rnapolis(
        barnaba_mapping.values(), rnapolis_mapping.keys()
    )
    residue_mapping = {
        barnaba: rnapolis
        for barnaba, rnapolis, _ in barnaba_to_rnapolis_mapping["pairs"]
    }

    base_pairs: List[BasePair] = []
    stackings: List[Stacking] = []
    other_interactions: List[OtherInteraction] = []

    for file_path, is_pairing, is_stacking in [
        (pairing_file, True, False),
        (stacking_file, False, True),
    ]:
        if file_path is None:
            continue

        logging.info(f"Processing barnaba file: {file_path}")

        with open(file_path) as f:
            content = f.read()

        breakpoint()

        for line in content.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            fields = line.split()
            if len(fields) < 3:
                continue

            res1_str, res2_str, interaction_str = fields[0], fields[1], fields[2]

            nt1 = _barnaba_get_residue(
                res1_str, barnaba_mapping, residue_mapping, rnapolis_mapping
            )
            nt2 = _barnaba_get_residue(
                res2_str, barnaba_mapping, residue_mapping, rnapolis_mapping
            )

            if not nt1 or not nt2:
                continue

            if is_pairing:
                try:
                    lw = _barnaba_get_leontis_westhof(interaction_str)
                    if lw:
                        base_pairs.append(BasePair(nt1, nt2, lw, None))
                    else:
                        other_interactions.append(OtherInteraction(nt1, nt2))
                except (KeyError, IndexError):
                    other_interactions.append(OtherInteraction(nt1, nt2))
            elif is_stacking:
                try:
                    topology_str = _BARNABA_STACKING_TOPOLOGIES.get(interaction_str)
                    if topology_str:
                        topology = StackingTopology[topology_str]
                        stackings.append(Stacking(nt1, nt2, topology))
                except KeyError:
                    logging.warning(
                        f"Unknown barnaba stacking topology: {interaction_str}"
                    )

    return BaseInteractions.from_structure3d(
        structure3d, base_pairs, stackings, [], [], other_interactions
    )


class MCAnnotateAdapter:
    # Represents state of parsing MC-Annotate result
    # Luckily every important part of file
    # begins with a unique sentence
    class ParseState(str, Enum):
        RESIDUES_INFORMATION = "Residue conformations"
        ADJACENT_STACKINGS = "Adjacent stackings"
        NON_ADJACENT_STACKINGS = "Non-Adjacent stackings"
        BASE_PAIRS_SECTION = "Base-pairs"
        SUMMARY_SECTION = "Number of"

    # This dictionary maps our model edges
    # to edge representation used by MC-Annotate
    EDGES: Dict[str, Tuple[str, ...]] = {
        "H": ("Hh", "Hw", "Bh", "C8"),
        "W": ("Wh", "Ww", "Ws"),
        "S": ("Ss", "Sw", "Bs"),
    }

    # Contains flatten EDGES values (in one touple)
    ALL_EDGES = sum(EDGES.values(), ())

    # Based on these tokens
    # BaseRiboseInteractions and BasePhosphateInteractions are created
    RIBOSE_ATOM = "O2'"
    PHOSPHATE_ATOM = "O2P"

    # Single hydrogen bond - for us it's OtherInteraction
    ONE_HBOND = "one_hbond"

    # Cis/trans tokens used by MC-Annotate
    CIS = "cis"
    TRANS = "trans"

    # Tokens used in PDB files
    ATOM = "ATOM"
    HETATM = "HETATM"

    # This regex is used to capture 6 groups of residues information:
    # (1) (2) (3) (4) (5) (6)
    # 1, 4 - chain IDs
    # 2, 5 - numbers
    # 3, 6 - icodes (or empty string if no icode)
    # Example - match and groups:
    # A-100.X-B200
    # ('A'), ('-100'), ('X'), ('B'), ('200'), ('')
    RESIDUE_REGEX = re.compile(
        r"'?(.)'?(-?[0-9]+)\.?([a-zA-Z]?)-'?(.)'?(-?[0-9]+)\.?([a-zA-Z]?)"
    )

    # Roman numerals used by Saenger
    # both in our model and MC-Annotate
    ROMAN_NUMERALS = ("I", "V", "X")

    # Positions of residues info in PDB files
    CHAIN_INDEX = 21
    NUMBER_INDEX = slice(22, 26)
    ICODE_INDEX = 26
    NAME_INDEX = slice(17, 20)

    def __init__(self) -> None:
        # Since names are not present in adjacent and non-adjacent stackings
        # we need save these values eariler
        self.names: Dict[str, str] = {}
        self.base_pairs: List[BasePair] = []
        self.stackings: List[Stacking] = []
        self.base_ribose_interactions: List[BaseRibose] = []
        self.base_phosphate_interactions: List[BasePhosphate] = []
        self.other_interactions: List[OtherInteraction] = []

    def classify_edge(self, edge_type: str) -> Optional[str]:
        for edge, edges in self.EDGES.items():
            if edge_type in edges:
                return edge
        logging.warning('Edge type "{type}" unknown')
        return None

    def get_residue(self, residue_info_list: Tuple[Union[str, Any], ...]) -> Residue:
        chain = residue_info_list[0]
        number = int(residue_info_list[1])

        if residue_info_list[2] == "":
            icode = None
            residue_info = f"{chain}{number}"
        else:
            icode = residue_info_list[2]
            residue_info = f"{chain}{number}.{icode}"

        return Residue(
            None, ResidueAuth(chain, number, icode, self.names[residue_info])
        )

    def get_residues(
        self, residues_info: str
    ) -> Tuple[Optional[Residue], Optional[Residue]]:
        regex_result = re.search(self.RESIDUE_REGEX, residues_info)
        if regex_result is None:
            logging.error("MC-Annotate regex failed: {residues_info}")
            return None, None
        residues_info_list = regex_result.groups()
        # Expects (chain1, number1, icode1, chain2, number2, icode2)
        if len(residues_info_list) != 6:
            logging.error(f"MC-Annotate regex failed for {residues_info}")
            return None, None
        residue_left = self.get_residue(residues_info_list[:3])
        residue_right = self.get_residue(residues_info_list[3:])
        return residue_left, residue_right

    def append_stacking(self, line: str, topology_position: int) -> None:
        splitted_line = line.split()
        topology_info = splitted_line[topology_position]
        residue_left, residue_right = self.get_residues(splitted_line[0])
        if residue_left is None or residue_right is None:
            logging.warning(f"Could not parse residues in line: {line}")
            return
        stacking = Stacking(
            residue_left, residue_right, StackingTopology[topology_info]
        )
        self.stackings.append(stacking)

    def get_ribose_interaction(
        self, residues: Tuple[Residue, Residue], token: str
    ) -> BaseRibose:
        # BasePair is preffered first so swap if necessary
        if token.split("/", 1)[0] == self.RIBOSE_ATOM:
            residue_left, residue_right = residues[1], residues[0]
        else:
            residue_left, residue_right = residues[0], residues[1]
        return BaseRibose(residue_left, residue_right, None)

    def get_phosphate_interaction(
        self, residues: Tuple[Residue, Residue], token: str
    ) -> BasePhosphate:
        # BasePair is preffered first so swap if necessary
        if token.split("/", 1)[0] == self.PHOSPHATE_ATOM:
            residue_left, residue_right = residues[1], residues[0]
        else:
            residue_left, residue_right = residues[0], residues[1]
        return BasePhosphate(residue_left, residue_right, None)

    def get_base_interaction(
        self,
        residues: Tuple[Residue, Residue],
        token: str,
        tokens: List[str],
    ) -> Optional[BasePair]:
        if self.CIS in tokens:
            cis_trans = "c"
        elif self.TRANS in tokens:
            cis_trans = "t"
        else:
            logging.warning(f"Cis/trans expected, but not present in {tokens}")
            return None

        # example saenger: XIX or XII,XIII (?)
        for potential_saenger_token in tokens:
            potential_saenger_without_comma = potential_saenger_token.split(",")[0]
            if all(
                char in self.ROMAN_NUMERALS for char in potential_saenger_without_comma
            ):
                saenger = Saenger[potential_saenger_without_comma]
                break
        else:
            saenger = None

        left_edge, right_edge = token.split("/", 1)
        leontis_westhof_left = self.classify_edge(left_edge)
        leontis_westohf_right = self.classify_edge(right_edge)

        if leontis_westhof_left is None or leontis_westohf_right is None:
            return None

        leontis_westhof = LeontisWesthof[
            f"{cis_trans}{leontis_westhof_left}{leontis_westohf_right}"
        ]
        residue_left, residue_right = residues
        return BasePair(residue_left, residue_right, leontis_westhof, saenger)

    def get_other_interaction(
        self, residues: Tuple[Residue, Residue]
    ) -> OtherInteraction:
        return OtherInteraction(residues[0], residues[1])

    def append_interactions(self, line: str) -> None:
        splitted_line = line.split()
        residues = self.get_residues(splitted_line[0])
        if residues[0] is None or residues[1] is None:
            logging.warning(f"Could not parse residues in line: {line}")
            return
        # Assumes that one pair can belong to every interaction type
        # no more than once!
        base_added, ribose_added, phosphate_added = False, False, False
        # example tokens: Ww/Ww pairing antiparallel cis XX
        tokens: List[str] = splitted_line[3:]

        # Special case
        # IF single hydrogen bond and base pairs only THEN
        # append to OtherIneraction list
        if self.ONE_HBOND in tokens:
            for token in tokens:
                if self.RIBOSE_ATOM in token or self.PHOSPHATE_ATOM in token:
                    break
            else:
                other_interaction = self.get_other_interaction(residues)
                self.other_interactions.append(other_interaction)
                return

        for token in tokens:
            if self.RIBOSE_ATOM in token and not ribose_added:
                # example token: Ss/O2'
                ribose_interaction = self.get_ribose_interaction(residues, token)
                self.base_ribose_interactions.append(ribose_interaction)
                ribose_added = True

            elif self.PHOSPHATE_ATOM in token and not phosphate_added:
                # example token: O2P/Bh
                phosphate_interaction = self.get_phosphate_interaction(residues, token)
                self.base_phosphate_interactions.append(phosphate_interaction)
                phosphate_added = True

            elif len(token.split("/", 1)) > 1:
                token_left, token_right = token.split("/", 1)
                tokens_in_edges = (
                    token_left in self.ALL_EDGES and token_right in self.ALL_EDGES
                )
                if tokens_in_edges and not base_added:
                    # example token_left: Ww | example token_right: Ws
                    base_pair_interaction = self.get_base_interaction(
                        residues, token, tokens
                    )
                    if base_pair_interaction is not None:
                        self.base_pairs.append(base_pair_interaction)
                    base_added = True

    def append_names(self, file_content: str) -> None:
        for line in file_content.splitlines():
            if line.startswith(self.ATOM) or line.startswith(self.HETATM):
                chain = line[self.CHAIN_INDEX].strip()
                number = line[self.NUMBER_INDEX].strip()
                icode = line[self.ICODE_INDEX].strip()
                name = line[self.NAME_INDEX].strip()
                residue_info = (
                    f"{chain}{number}" if icode == "" else f"{chain}{number}.{icode}"
                )
                self.names[residue_info] = name

    def analyze_by_mc_annotate(
        self, pdb_content: str, mc_result: str, **_: Dict[str, Any]
    ) -> BaseInteractions:
        self.append_names(pdb_content)
        current_state = None

        for line in mc_result.splitlines():
            for state in self.ParseState:
                if line.startswith(state.value):
                    current_state = state
                    break
            # Loop ended without break - parse file
            else:
                if current_state == self.ParseState.RESIDUES_INFORMATION:
                    # example line: X7.H : G C3p_endo anti
                    # Skip residues information - meaningless information
                    pass
                elif current_state == self.ParseState.ADJACENT_STACKINGS:
                    # example line: X4.E-X5.F : adjacent_5p upward
                    self.append_stacking(line, 3)
                elif current_state == self.ParseState.NON_ADJACENT_STACKINGS:
                    # example line: Y40.M-Y67.N : inward pairing
                    self.append_stacking(line, 2)
                elif current_state == self.ParseState.BASE_PAIRS_SECTION:
                    # example line: Y38.K-Y51.X : A-U Ww/Ww pairing antiparallel cis XX
                    self.append_interactions(line)
                elif current_state == self.ParseState.SUMMARY_SECTION:
                    # example line: Number of non adjacent stackings = 26
                    # Skip summary section - meaningless information
                    pass

        return (
            self.base_pairs,
            self.stackings,
            self.base_ribose_interactions,
            self.base_phosphate_interactions,
            self.other_interactions,
        )


def parse_mcannotate_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse mc-annotate output and convert to BaseInteractions.
    This function expects a file with mc-annotate stdout and a PDB file.
    """
    stdout_file = None
    structure_file = None
    for file_path in file_paths:
        if os.path.basename(file_path).endswith("stdout.txt"):
            stdout_file = file_path
        elif file_path.endswith(".pdb"):
            structure_file = file_path

    if not stdout_file:
        logging.warning("No stdout.txt file found for mc-annotate.")
        return BaseInteractions([], [], [], [], [])

    if not structure_file:
        logging.warning("No PDB file found for mc-annotate.")
        return BaseInteractions([], [], [], [], [])

    logging.info(f"Processing mc-annotate stdout file: {stdout_file}")
    logging.info(f"Using structure file for residue names: {structure_file}")

    try:
        with open(stdout_file, "r") as f:
            mc_result = f.read()
        with open(structure_file, "r") as f:
            pdb_content = f.read()
    except Exception as e:
        logging.warning(f"Could not read input files for mc-annotate: {e}")
        return BaseInteractions([], [], [], [], [])

    adapter = MCAnnotateAdapter()
    (
        base_pairs,
        stackings,
        base_ribose_interactions,
        base_phosphate_interactions,
        other_interactions,
    ) = adapter.analyze_by_mc_annotate(pdb_content, mc_result)

    return BaseInteractions.from_structure3d(
        structure3d,
        base_pairs,
        stackings,
        base_ribose_interactions,
        base_phosphate_interactions,
        other_interactions,
    )


def parse_external_output(
    file_paths: List[str], tool: ExternalTool, structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse the output from an external tool (FR3D, DSSR, etc.) and convert it to BaseInteractions.

    Args:
        file_paths: List of paths to external tool output files
        tool: The external tool that generated the output
        structure3d: The 3D structure parsed from PDB/mmCIF

    Returns:
        BaseInteractions object containing the interactions found by the external tool
    """
    if tool == ExternalTool.FR3D:
        return parse_fr3d_output(file_paths, structure3d)
    elif tool == ExternalTool.DSSR:
        return parse_dssr_output(file_paths, structure3d)
    elif tool == ExternalTool.MAXIT:
        return parse_maxit_output(file_paths, structure3d)
    elif tool == ExternalTool.BPNET:
        return parse_bpnet_output(file_paths, structure3d)
    elif tool == ExternalTool.RNAVIEW:
        return parse_rnaview_output(file_paths, structure3d)
    elif tool == ExternalTool.BARNABA:
        return parse_barnaba_output(file_paths, structure3d)
    elif tool == ExternalTool.MCANNOTATE:
        return parse_mcannotate_output(file_paths, structure3d)
    else:
        raise ValueError(f"Unsupported external tool: {tool}")


def parse_fr3d_output(
    file_paths: List[str], structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse FR3D output files and convert to BaseInteractions.

    Args:
        file_paths: List of paths to FR3D output files containing basepair, stacking,
                   and backbone interactions

    Returns:
        BaseInteractions object containing the interactions found by FR3D
    """
    # Initialize the interaction data dictionary
    interactions_data: Dict[str, list] = {
        "base_pairs": [],
        "stackings": [],
        "base_ribose_interactions": [],
        "base_phosphate_interactions": [],
        "other_interactions": [],
    }

    # Process each input file
    for file_path in file_paths:
        logging.info(f"Processing FR3D file: {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Process every non-empty, non-comment line
                _process_interaction_line(line, interactions_data)

    # Return a BaseInteractions object with all the processed interactions
    return BaseInteractions.from_structure3d(
        structure3d,
        interactions_data["base_pairs"],
        interactions_data["stackings"],
        interactions_data["base_ribose_interactions"],
        interactions_data["base_phosphate_interactions"],
        interactions_data["other_interactions"],
    )


def process_external_tool_output(
    structure3d: Structure3D,
    external_file_paths: List[str],
    tool: ExternalTool,
    input_file_path: str,
    find_gaps: bool = False,
) -> Tuple[Structure2D, Mapping2D3D]:  # Added Mapping2D3D to return tuple
    """
    Process external tool output and create a secondary structure representation.

    This function can be used from other code to process external tool outputs
    and get a Structure2D object with the secondary structure information.

    Args:
        structure3d: The 3D structure parsed from PDB/mmCIF
        external_file_paths: List of paths to external tool output files (empty for MAXIT)
        tool: The external tool that generated the output (FR3D, DSSR, etc.)
        input_file_path: Path to the input file (used when external_file_paths is empty)
        find_gaps: Whether to detect gaps in the structure

    Returns:
        A tuple containing the Structure2D object and the Mapping2D3D object.
    """
    # Parse external tool output
    if not external_file_paths:
        # For MAXIT or when no external files are provided, use the input file
        file_paths_to_process = [input_file_path]
    elif tool == ExternalTool.MCANNOTATE or tool == ExternalTool.RNAVIEW:
        # MC-Annotate and RNAView requires both the stdout and the PDB file
        file_paths_to_process = external_file_paths + [input_file_path]
    else:
        # Process all external files
        file_paths_to_process = external_file_paths

    base_interactions = parse_external_output(file_paths_to_process, tool, structure3d)

    # Extract secondary structure using the external tool's interactions
    return structure3d.extract_secondary_structure(base_interactions, find_gaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument(
        "external_files",
        nargs="*",
        help="Path(s) to external tool output file(s) (FR3D, DSSR, etc.)",
    )
    parser.add_argument(
        "--tool",
        choices=[t.value for t in ExternalTool],
        help="External tool that generated the output file (auto-detected if not specified)",
    )
    parser.add_argument(
        "-f",
        "--find-gaps",
        action="store_true",
        help="(optional) if set, the program will detect gaps and break the PDB chain into two or more strands",
    )
    add_common_output_arguments(parser)
    # The --inter-stem-csv and --stems-csv arguments are now added by add_common_output_arguments
    args = parser.parse_args()

    file = handle_input_file(args.input)
    structure3d = read_3d_structure(file, None)

    # Auto-detect tool if not specified
    if args.tool is not None:
        tool = ExternalTool(args.tool)
    else:
        tool = auto_detect_tool(args.external_files)
        logging.info(f"Auto-detected tool: {tool.value}")

    # Process external tool output files and get secondary structure
    # Always call process_external_tool_output, even for MAXIT (empty external files)
    structure2d, mapping = process_external_tool_output(
        structure3d,
        args.external_files,
        tool,
        args.input,
        args.find_gaps,
    )

    handle_output_arguments(args, structure2d, mapping, args.input)


if __name__ == "__main__":
    main()
