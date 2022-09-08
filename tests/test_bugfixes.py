from rnapolis.annotator import extract_secondary_structure
from rnapolis.common import ResidueLabel
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


# in 1E7K there is a break in a chain, so it should be recognized as separate strands
def test_1E7K():
    with open("tests/1E7K_1_C.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert len(mapping.strands_sequences) == 2


# in 1DFU the adjacent U and G seem to be base-pair like if you do not take into account angles
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


# in 4WTI the first residue has only O3' atom and so is not considered a nucleotide
def test_4WTI():
    with open("tests/4WTI_1_T-P.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert mapping.dot_bracket == ">strand_T\nCGG\n.((\n>strand_P\nCC\n))"


# in 1HMH the bases are oriented in 45 degrees and it caused the program to identify invalid base pair
def test_1HMH():
    with open("tests/1HMH_1_E.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert mapping.dot_bracket == ">strand_E\nUG\n.."


# 2HY9 has a quadruplex
def test_2HY9():
    with open("tests/2HY9.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)

    # tract 1
    g4 = structure3d.find_residue(ResidueLabel("A", 4, "DG"), None)
    g5 = structure3d.find_residue(ResidueLabel("A", 5, "DG"), None)
    g6 = structure3d.find_residue(ResidueLabel("A", 6, "DG"), None)

    # tract 2
    g10 = structure3d.find_residue(ResidueLabel("A", 10, "DG"), None)
    g11 = structure3d.find_residue(ResidueLabel("A", 11, "DG"), None)
    g12 = structure3d.find_residue(ResidueLabel("A", 12, "DG"), None)

    # tract 3
    g16 = structure3d.find_residue(ResidueLabel("A", 16, "DG"), None)
    g17 = structure3d.find_residue(ResidueLabel("A", 17, "DG"), None)
    g18 = structure3d.find_residue(ResidueLabel("A", 18, "DG"), None)

    # tract 4
    g22 = structure3d.find_residue(ResidueLabel("A", 22, "DG"), None)
    g23 = structure3d.find_residue(ResidueLabel("A", 23, "DG"), None)
    g24 = structure3d.find_residue(ResidueLabel("A", 24, "DG"), None)

    assert all(
        nt is not None
        for nt in [g4, g5, g6, g10, g11, g12, g16, g17, g18, g22, g23, g24]
    )

    # tetrad 1
    assert {g10, g22}.issubset(mapping.base_pair_graph[g4])  # type: ignore
    assert {g10, g22}.issubset(mapping.base_pair_graph[g18])  # type: ignore
    assert {g4, g18}.issubset(mapping.base_pair_graph[g10])  # type: ignore
    assert {g4, g18}.issubset(mapping.base_pair_graph[g22])  # type: ignore

    # tetrad 2
    assert {g11, g23}.issubset(mapping.base_pair_graph[g5])  # type: ignore
    assert {g11, g23}.issubset(mapping.base_pair_graph[g17])  # type: ignore
    assert {g5, g17}.issubset(mapping.base_pair_graph[g11])  # type: ignore
    assert {g5, g17}.issubset(mapping.base_pair_graph[g23])  # type: ignore

    # tetrad 3
    assert {g12, g24}.issubset(mapping.base_pair_graph[g6])  # type: ignore
    assert {g12, g24}.issubset(mapping.base_pair_graph[g16])  # type: ignore
    assert {g6, g16}.issubset(mapping.base_pair_graph[g12])  # type: ignore
    assert {g6, g16}.issubset(mapping.base_pair_graph[g24])  # type: ignore
