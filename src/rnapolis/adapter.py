#! /usr/bin/env python
import argparse
import logging
import os
from enum import Enum
from typing import List, Tuple

from rnapolis.adapters.barnaba import parse_barnaba_output
from rnapolis.adapters.bpnet import parse_bpnet_output
from rnapolis.adapters.dssr import parse_dssr_output
from rnapolis.adapters.fr3d import parse_fr3d_output
from rnapolis.adapters.maxit import parse_maxit_output
from rnapolis.adapters.mc_annotate import parse_mcannotate_output
from rnapolis.adapters.rnaview import parse_rnaview_output
from rnapolis.annotator import (
    add_common_output_arguments,
    handle_output_arguments,
)
from rnapolis.common import BaseInteractions, Structure2D
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    Mapping2D3D,
    Structure3D,
)
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
