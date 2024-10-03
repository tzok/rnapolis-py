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
