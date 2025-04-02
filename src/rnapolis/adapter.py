#! /usr/bin/env python
import argparse
import csv
import logging
import os
from enum import Enum
from typing import IO, Dict, List, Optional, Tuple

import orjson

from rnapolis.common import BaseInteractions, BpSeq, Structure2D
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D, Structure3D
from rnapolis.util import handle_input_file


class ExternalTool(Enum):
    FR3D = "fr3d"
    DSSR = "dssr"


logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


def parse_external_output(
    file_path: str, tool: ExternalTool, structure3d: Structure3D
) -> BaseInteractions:
    """
    Parse the output from an external tool (FR3D, DSSR, etc.) and convert it to BaseInteractions.
    
    Args:
        file_path: Path to the external tool output file
        tool: The external tool that generated the output
        structure3d: The 3D structure parsed from PDB/mmCIF
        
    Returns:
        BaseInteractions object containing the interactions found by the external tool
    """
    # This is a placeholder - actual implementation will be added later
    if tool == ExternalTool.FR3D:
        # TODO: Implement FR3D output parsing
        raise NotImplementedError("FR3D output parsing not yet implemented")
    elif tool == ExternalTool.DSSR:
        # TODO: Implement DSSR output parsing
        raise NotImplementedError("DSSR output parsing not yet implemented")
    else:
        raise ValueError(f"Unsupported external tool: {tool}")


def extract_secondary_structure_from_external(
    tertiary_structure: Structure3D,
    base_interactions: BaseInteractions,
    model: Optional[int] = None,
    find_gaps: bool = False,
    all_dot_brackets: bool = False,
) -> Tuple[Structure2D, List[str]]:
    """
    Create a secondary structure representation using interactions from an external tool.
    
    Args:
        tertiary_structure: The 3D structure parsed from PDB/mmCIF
        base_interactions: Interactions parsed from external tool output
        model: Model number to use (if None, use all models)
        find_gaps: Whether to detect gaps in the structure
        all_dot_brackets: Whether to return all possible dot-bracket notations
        
    Returns:
        A tuple containing the Structure2D object and a list of dot-bracket notations
    """
    mapping = Mapping2D3D(
        tertiary_structure,
        base_interactions.basePairs,
        base_interactions.stackings,
        find_gaps,
    )
    stems, single_strands, hairpins, loops = mapping.bpseq.elements
    structure2d = Structure2D(
        base_interactions,
        str(mapping.bpseq),
        mapping.dot_bracket,
        mapping.extended_dot_bracket,
        stems,
        single_strands,
        hairpins,
        loops,
    )
    if all_dot_brackets:
        return structure2d, mapping.all_dot_brackets
    else:
        return structure2d, [structure2d.dotBracket]


def write_json(path: str, structure2d: BaseInteractions):
    with open(path, "wb") as f:
        f.write(orjson.dumps(structure2d))


def write_csv(path: str, structure2d: Structure2D):
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["nt1", "nt2", "type", "classification-1", "classification-2"])
        for base_pair in structure2d.baseInteractions.basePairs:
            writer.writerow(
                [
                    base_pair.nt1.full_name,
                    base_pair.nt2.full_name,
                    "base pair",
                    base_pair.lw.value,
                    (
                        base_pair.saenger.value or ""
                        if base_pair.saenger is not None
                        else ""
                    ),
                ]
            )
        for stacking in structure2d.baseInteractions.stackings:
            writer.writerow(
                [
                    stacking.nt1.full_name,
                    stacking.nt2.full_name,
                    "stacking",
                    stacking.topology.value if stacking.topology is not None else "",
                    "",
                ]
            )
        for base_phosphate in structure2d.baseInteractions.basePhosphateInteractions:
            writer.writerow(
                [
                    base_phosphate.nt1.full_name,
                    base_phosphate.nt2.full_name,
                    "base-phosphate interaction",
                    base_phosphate.bph.value if base_phosphate.bph is not None else "",
                    "",
                ]
            )
        for base_ribose in structure2d.baseInteractions.baseRiboseInteractions:
            writer.writerow(
                [
                    base_ribose.nt1.full_name,
                    base_ribose.nt2.full_name,
                    "base-ribose interaction",
                    base_ribose.br.value if base_ribose.br is not None else "",
                    "",
                ]
            )
        for other in structure2d.baseInteractions.otherInteractions:
            writer.writerow(
                [
                    other.nt1.full_name,
                    other.nt2.full_name,
                    "other interaction",
                    "",
                    "",
                ]
            )


def write_bpseq(path: str, bpseq: BpSeq):
    with open(path, "w") as f:
        f.write(str(bpseq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to PDB or mmCIF file")
    parser.add_argument(
        "--external",
        required=True,
        help="Path to external tool output file (FR3D, DSSR, etc.)",
    )
    parser.add_argument(
        "--tool",
        choices=[t.value for t in ExternalTool],
        required=True,
        help="External tool that generated the output file",
    )
    parser.add_argument(
        "-a",
        "--all-dot-brackets",
        action="store_true",
        help="(optional) print all dot-brackets, not only optimal one (exclusive with -e/--extended)",
    )
    parser.add_argument("-b", "--bpseq", help="(optional) path to output BPSEQ file")
    parser.add_argument("-c", "--csv", help="(optional) path to output CSV file")
    parser.add_argument(
        "-j",
        "--json",
        help="(optional) path to output JSON file",
    )
    parser.add_argument(
        "-e",
        "--extended",
        action="store_true",
        help="(optional) if set, the program will print extended secondary structure to the standard output",
    )
    parser.add_argument(
        "-f",
        "--find-gaps",
        action="store_true",
        help="(optional) if set, the program will detect gaps and break the PDB chain into two or more strands",
    )
    parser.add_argument("-d", "--dot", help="(optional) path to output DOT file")
    args = parser.parse_args()

    file = handle_input_file(args.input)
    structure3d = read_3d_structure(file, None)
    
    # Parse external tool output
    external_tool = ExternalTool(args.tool)
    base_interactions = parse_external_output(args.external, external_tool, structure3d)
    
    # Extract secondary structure using the external tool's interactions
    structure2d, dot_brackets = extract_secondary_structure_from_external(
        structure3d, base_interactions, None, args.find_gaps, args.all_dot_brackets
    )

    if args.csv:
        write_csv(args.csv, structure2d)

    if args.json:
        write_json(args.json, structure2d)

    if args.bpseq:
        write_bpseq(args.bpseq, structure2d.bpseq)

    if args.extended:
        print(structure2d.extendedDotBracket)
    elif args.all_dot_brackets:
        for dot_bracket in dot_brackets:
            print(dot_bracket)
    else:
        print(structure2d.dotBracket)

    if args.dot:
        print(BpSeq.from_string(structure2d.bpseq).graphviz)


if __name__ == "__main__":
    main()
