from pathlib import Path

import pytest

from rnapolis.adapter import ExternalTool, process_external_tool_output
from rnapolis.common import BaseInteractions
from rnapolis.parser import read_3d_structure
from rnapolis.structure_index import read_structure_index
from rnapolis.tertiary import extract_secondary_structure
from rnapolis.util import handle_input_file

TEST_DIR = Path(__file__).parent


def _residue_signature(structure):
    return [
        (
            residue.label,
            residue.auth,
            residue.model,
            residue.standard_residue_name,
            residue.one_letter_name,
            residue.is_nucleotide,
        )
        for residue in structure.residues
    ]


@pytest.mark.parametrize(
    "path",
    [
        TEST_DIR / "1A4D_1_A-B.cif",
        TEST_DIR / "2HY9.cif",
        TEST_DIR / "5it9.cif",
        TEST_DIR / "1a9n.cif",
        TEST_DIR / "4qln.pdb",
        TEST_DIR / "488d.pdb",
    ],
)
def test_structure_index_matches_full_structure(path):
    """The lightweight index preserves the full parser's residue-level view."""
    full = read_3d_structure(handle_input_file(str(path)), None)
    index = read_structure_index(handle_input_file(str(path)), None)

    assert _residue_signature(index) == _residue_signature(full)


def test_structure_index_find_residue():
    path = TEST_DIR / "1A4D_1_A-B.cif"
    index = read_structure_index(handle_input_file(str(path)), None)

    assert index.find_residue(None, None) is None
    residue = index.residues[0]
    assert index.find_residue(residue.label, None) is residue
    assert index.find_residue(None, residue.auth) is residue


def test_structure_index_gzip_matches_plain_cif():
    plain = read_structure_index(
        handle_input_file(str(TEST_DIR / "1A4D_1_A-B.cif")), None
    )
    compressed = read_structure_index(
        handle_input_file(str(TEST_DIR / "1A4D_1_A-B.cif.gz")), None
    )

    assert _residue_signature(compressed) == _residue_signature(plain)


def test_structure_index_rejects_gap_aware_mapping():
    path = TEST_DIR / "1A4D_1_A-B.cif"
    index = read_structure_index(handle_input_file(str(path)), None)

    with pytest.raises(TypeError, match="gap-aware"):
        extract_secondary_structure(
            index,
            BaseInteractions([], [], [], [], []),
            find_gaps=True,
        )


@pytest.mark.parametrize(
    ("tool", "external_file"),
    [
        (ExternalTool.FR3D, "1A4D_1_A-B-basepair_detail.txt"),
        (ExternalTool.BPNET, "1A4D_1_A-B-input_basepair.json"),
        (ExternalTool.RNAVIEW, "1A4D_1_A-B-input.cif.out"),
        (ExternalTool.MAXIT, "1A4D_1_A-B-output.cif"),
        (ExternalTool.DSSR, "1A4D_1_A-B-dssr.json"),
    ],
)
def test_adapter_output_matches_full_structure(tool, external_file):
    """Adapters produce identical dot-bracket output from full and indexed data."""
    cif_path = TEST_DIR / "1A4D_1_A-B.cif"
    external_path = TEST_DIR / external_file

    full = read_3d_structure(handle_input_file(str(cif_path)), None)
    index = read_structure_index(handle_input_file(str(cif_path)), None)

    full_mapping = process_external_tool_output(
        full, [str(external_path)], tool, str(cif_path)
    )[1]
    index_mapping = process_external_tool_output(
        index, [str(external_path)], tool, str(cif_path)
    )[1]

    assert index_mapping.dot_bracket == full_mapping.dot_bracket
    assert index_mapping.bpseq == full_mapping.bpseq


def test_dnatco_adapter_output_matches_full_structure():
    """DNATCO conversion is independent of the full 3D representation."""
    cif_path = TEST_DIR / "dnatco/1ehz/1ehz.cif"
    dnatco_path = TEST_DIR / "dnatco/1ehz/1ehz.csv"

    full = read_3d_structure(handle_input_file(str(cif_path)), None)
    index = read_structure_index(handle_input_file(str(cif_path)), None)

    full_mapping = process_external_tool_output(
        full, [str(dnatco_path)], ExternalTool.DNATCO, str(cif_path)
    )[1]
    index_mapping = process_external_tool_output(
        index, [str(dnatco_path)], ExternalTool.DNATCO, str(cif_path)
    )[1]

    assert index_mapping.dot_bracket == full_mapping.dot_bracket
    assert index_mapping.bpseq == full_mapping.bpseq
