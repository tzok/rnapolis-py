import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    ResidueAuth,
    Saenger,
    Stacking,
)
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary import Structure3D
from rnapolis.tertiary_v2 import Structure as StructureV2


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
            residues_from_input = (
                _rnaview_append_residues_from_input_using_rnaview_indexing(
                    input_content, input_type
                )
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
