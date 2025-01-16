import gzip

from rnapolis.parser import read_3d_structure


def test_nucleic_acid_only():
    with open("tests/184D.cif") as f:
        structure3d = read_3d_structure(f, 1, nucleic_acid_only=False)
    assert len(structure3d.residues) == 114

    with open("tests/184D.cif") as f:
        structure3d = read_3d_structure(f, 1, nucleic_acid_only=True)
    assert len(structure3d.residues) == 14


def test_1ato():
    with open("tests/1ATO.pdb") as f:
        structure3d = read_3d_structure(f)
    sequence = "".join([residue.one_letter_name for residue in structure3d.residues])
    assert sequence == "GGCACCUCCUCGCGGUGCC"


def test_4qln_no_duplicate_atoms():
    for ext in (".pdb", ".cif"):
        with open(f"tests/4qln{ext}") as f:
            structure3d = read_3d_structure(f)

        chain_a = [r for r in structure3d.residues if r.auth.chain == "A"]
        residues_to_check = [r for r in chain_a if r.auth.number in (18, 19, 20)]

        for residue in residues_to_check:
            atom_names = [atom.name for atom in residue.atoms]
            assert len(atom_names) == len(
                set(atom_names)
            ), f"Duplicate atoms found in residue {residue.auth}"


def test_1gid():
    expected_sequence = "GAAUUGCGGGAAAGGGGUCAACAGCCGUUCAGUACCAAGUCUCAGGGGAAACUUUGAGAUGGCCUUGCAAAGGGUAUGGUAAUAAGCUGACGGACAUGGUCCUAACCACGCAGCCAAGUCCUAAGUCAACAGAUCUUCUGUUGAUAUGGAUGCAGUUC"

    with gzip.open("tests/1gid.cif.gz", "rt") as f:
        structure3d = read_3d_structure(f, nucleic_acid_only=True)

    residues_a = [r for r in structure3d.residues if r.auth.chain == "A"]
    residues_b = [r for r in structure3d.residues if r.auth.chain == "B"]
    assert len(residues_a) == len(expected_sequence)
    assert len(residues_b) == len(expected_sequence)

    actual_sequence_a = "".join([residue.one_letter_name for residue in residues_a])
    actual_sequence_b = "".join([residue.one_letter_name for residue in residues_b])
    assert actual_sequence_a == expected_sequence
    assert actual_sequence_b == expected_sequence
