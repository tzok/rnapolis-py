from rnapolis.annotator import extract_secondary_structure
from rnapolis.common import ResidueLabel
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


def test_1E7K():
    with open("tests/1E7K_1_C.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert len(mapping.strands_sequences) == 2


def test_1DFU():
    with open("tests/1DFU_1_M-N.cif") as f:
        structure3d = read_3d_structure(f, 1)

    b1u = structure3d.find_residue(ResidueLabel("B", 1, "U"), None)
    assert b1u is not None

    b2g = structure3d.find_residue(ResidueLabel("B", 2, "G"), None)
    assert b2g is not None

    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert b2g not in mapping.base_pair_graph[b1u]
    assert b1u not in mapping.base_pair_graph[b2g]


def test_4WTI():
    with open("tests/4WTI_1_T-P.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert mapping.dot_bracket == ">strand_T\nCGG\n.((\n>strand_P\nCC\n))"
