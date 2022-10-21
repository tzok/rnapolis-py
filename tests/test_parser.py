from rnapolis.parser import read_3d_structure


def test_nucleic_acid_only():
    with open("tests/184D.cif") as f:
        structure3d = read_3d_structure(f, 1, nucleic_acid_only=False)
    assert len(structure3d.residues) == 114

    with open("tests/184D.cif") as f:
        structure3d = read_3d_structure(f, 1, nucleic_acid_only=True)
    assert len(structure3d.residues) == 14
