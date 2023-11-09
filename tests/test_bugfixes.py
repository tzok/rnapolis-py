from rnapolis.annotator import extract_base_interactions
from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


# in 1E7K there is a break in a chain, so it should be recognized as separate strands
def test_1E7K():
    with open("tests/1E7K_1_C.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interaction = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interaction.basePairs, base_interaction.stackings, True
    )
    assert len(mapping.strands_sequences) == 2


# in 1DFU the adjacent U and G seem to be base-pair like if you do not take into account angles
def test_1DFU():
    with open("tests/1DFU_1_M-N.cif") as f:
        structure3d = read_3d_structure(f, 1)

    b1u = structure3d.find_residue(ResidueLabel("B", 1, "U"), None)
    assert b1u is not None

    b2g = structure3d.find_residue(ResidueLabel("B", 2, "G"), None)
    assert b2g is not None

    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )
    assert b2g not in mapping.base_pair_graph[b1u]
    assert b1u not in mapping.base_pair_graph[b2g]


# in 4WTI the first residue has only O3' atom and so is not considered a nucleotide
def test_4WTI():
    with open("tests/4WTI_1_T-P.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )
    assert mapping.dot_bracket == ">strand_T\nCGG\n.((\n>strand_P\nCC\n))"


# in 1HMH the bases are oriented in 45 degrees and it caused the program to identify invalid base pair
def test_1HMH():
    with open("tests/1HMH_1_E.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )
    assert mapping.dot_bracket == ">strand_E\nUG\n.."


# in 6INQ the residues T.DC0 and N.DG0 were not found by RNApolis
def test_6INQ():
    with open("tests/6INQ.cif") as f:
        structure3d = read_3d_structure(f, 1)
    assert structure3d.find_residue(ResidueLabel("N", 73, "DC"), None) is not None
    assert structure3d.find_residue(None, ResidueAuth("T", 0, None, "DC")) is not None
    assert structure3d.find_residue(ResidueLabel("O", 126, "DG"), None) is not None
    assert structure3d.find_residue(None, ResidueAuth("N", 0, None, "DG")) is not None
