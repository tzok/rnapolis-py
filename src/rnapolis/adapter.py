#! /usr/bin/env python
import argparse
import csv
import logging
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import orjson

from rnapolis.common import (
    BR,
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BPh,
    BpSeq,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    ResidueAuth,
    Stacking,
    StackingTopology,
    Structure2D,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D, Structure3D
from rnapolis.util import handle_input_file


class ExternalTool(Enum):
    FR3D = "fr3d"
    DSSR = "dssr"


logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


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
                return residue
        logging.warning(f"Failed to find residue {nt_id}")
    return None


def match_dssr_lw(lw: Optional[str]) -> Optional[LeontisWesthof]:
    return LeontisWesthof[lw] if lw in dir(LeontisWesthof) else None


def parse_dssr_output(
    file_path: str, structure3d: Structure3D, model: Optional[int] = None
) -> BaseInteractions:
    """
    Parse DSSR JSON output and convert to BaseInteractions.

    Args:
        file_path: Path to DSSR JSON output file
        structure3d: The 3D structure parsed from PDB/mmCIF
        model: Model number to use (if None, use first model)

    Returns:
        BaseInteractions object containing the interactions found by DSSR
    """
    base_pairs: List[BasePair] = []
    stackings: List[Stacking] = []

    with open(file_path) as f:
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

    return BaseInteractions(base_pairs, stackings, [], [], [])


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
    if tool == ExternalTool.FR3D:
        return parse_fr3d_output(file_path)
    elif tool == ExternalTool.DSSR:
        return parse_dssr_output(file_path, structure3d)
    else:
        raise ValueError(f"Unsupported external tool: {tool}")


def parse_fr3d_output(file_path: str) -> BaseInteractions:
    """
    Parse FR3D output file and convert to BaseInteractions.

    Args:
        file_path: Path to a concatenated FR3D output file containing basepair, stacking,
                  and backbone interactions

    Returns:
        BaseInteractions object containing the interactions found by FR3D
    """
    # Initialize the interaction data dictionary
    interactions_data = {
        "base_pairs": [],
        "stackings": [],
        "base_ribose_interactions": [],
        "base_phosphate_interactions": [],
        "other_interactions": [],
    }

    # Process the concatenated file
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Process every non-empty, non-comment line
            _process_interaction_line(line, interactions_data)

    # Return a BaseInteractions object with all the processed interactions
    return BaseInteractions(
        interactions_data["base_pairs"],
        interactions_data["stackings"],
        interactions_data["base_ribose_interactions"],
        interactions_data["base_phosphate_interactions"],
        interactions_data["other_interactions"],
    )


def process_external_tool_output(
    structure3d: Structure3D,
    external_file_path: str,
    tool: ExternalTool,
    model: Optional[int] = None,
    find_gaps: bool = False,
    all_dot_brackets: bool = False,
) -> Tuple[Structure2D, List[str]]:
    """
    Process external tool output and create a secondary structure representation.

    This function can be used from other code to process external tool outputs
    and get a Structure2D object with the secondary structure information.

    Args:
        structure3d: The 3D structure parsed from PDB/mmCIF
        external_file_path: Path to the external tool output file
        tool: The external tool that generated the output (FR3D, DSSR, etc.)
        model: Model number to use (if None, use first model)
        find_gaps: Whether to detect gaps in the structure
        all_dot_brackets: Whether to return all possible dot-bracket notations

    Returns:
        A tuple containing the Structure2D object and a list of dot-bracket notations
    """
    # Parse external tool output
    base_interactions = parse_external_output(external_file_path, tool, structure3d)

    # Extract secondary structure using the external tool's interactions
    return extract_secondary_structure_from_external(
        structure3d, base_interactions, model, find_gaps, all_dot_brackets
    )


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

    # Process external tool output and get secondary structure
    structure2d, dot_brackets = process_external_tool_output(
        structure3d,
        args.external,
        ExternalTool(args.tool),
        None,
        args.find_gaps,
        args.all_dot_brackets,
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
