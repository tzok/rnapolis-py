#! /usr/bin/env python
import argparse
import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

from rnapolis.annotator import (
    add_common_output_arguments,
    handle_output_arguments,
)
from rnapolis.common import Structure2D
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    Mapping2D3D,
    Structure3D,
)
from rnapolis.util import handle_input_file

from rnapolis.adapters.fr3d import parse_fr3d_output
from rnapolis.adapters.dssr import parse_dssr_output
from rnapolis.adapters.maxit import parse_maxit_output
from rnapolis.adapters.bpnet import parse_bpnet_output
from rnapolis.adapters.rnaview import parse_rnaview_output
from rnapolis.adapters.barnaba import parse_barnaba_output
from rnapolis.adapters.mc_annotate import parse_mcannotate_output


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


# RNAView regex pattern from the reference implementation

# RNAView tokens


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
