from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


def test_1E7K():
    with open("tests/1E7K_1_C.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d)
    mapping = Mapping2D3D(structure3d, structure2d)
    assert len(mapping.strands_sequences) == 2
