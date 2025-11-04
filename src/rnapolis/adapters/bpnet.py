import logging
from typing import List

import orjson

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    LeontisWesthof,
    OtherInteraction,
    Residue,
    ResidueAuth,
    Stacking,
)
from rnapolis.tertiary import Structure3D


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
