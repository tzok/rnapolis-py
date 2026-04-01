import math

from conftest import make_residue3d

from rnapolis.common import Molecule, ResidueAuth, ResidueLabel
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Atom, Residue3D, torsion_angle


def test_torsion_angle():
    a1 = Atom(None, None, None, 1, "P", 50.63, 49.73, 50.57, None)
    a2 = Atom(None, None, None, 1, "O5'", 50.16, 49.14, 52.02, None)
    a3 = Atom(None, None, None, 1, "C5'", 50.22, 49.95, 53.21, None)
    a4 = Atom(None, None, None, 1, "C4'", 50.97, 49.23, 54.31, None)
    assert math.isclose(
        math.degrees(torsion_angle(a1, a2, a3, a4)), -127.83976634524326
    )


def test_nucleobase_atoms():
    with open("tests/1E7K_1_C.cif") as f:
        structure3d = read_3d_structure(f)
    for residue in structure3d.residues:
        assert residue.has_all_nucleobase_heavy_atoms

    with open("tests/1E7K_1_C_modified.cif") as f:
        structure3d = read_3d_structure(f)
    assert all(
        not residue.has_all_nucleobase_heavy_atoms for residue in structure3d.residues
    )


def test_molecule_type_rna_by_name():
    """Standard RNA residue names should be classified by name, no atoms needed."""
    for name in ("A", "C", "G", "U", "I"):
        residue = make_residue3d(name, [])
        assert residue.molecule_type == Molecule.RNA, f"{name} should be RNA"


def test_molecule_type_dna_by_name():
    """Standard DNA residue names should be classified by name, no atoms needed."""
    for name in ("DA", "DC", "DG", "DT", "DU", "DI"):
        residue = make_residue3d(name, [])
        assert residue.molecule_type == Molecule.DNA, f"{name} should be DNA"


def test_molecule_type_atom_fallback_rna():
    """When the residue name is non-standard, O2' presence means RNA."""
    residue = make_residue3d("UNK", ["O2'"])
    assert residue.molecule_type == Molecule.RNA


def test_molecule_type_atom_fallback_dna():
    """When the residue name is non-standard, sugar atoms without O2' means DNA."""
    residue = make_residue3d("UNK", ["C1'", "C2'", "C3'", "C4'", "O4'"])
    assert residue.molecule_type == Molecule.DNA


def test_molecule_type_atom_fallback_other():
    """When name is non-standard and atoms are inconclusive, classify as Other."""
    residue = make_residue3d("UNK", ["N", "CA", "C", "O"])
    assert residue.molecule_type == Molecule.Other


def test_molecule_type_modres_rna():
    """Modified residue with standard_residue_name resolved to RNA name should be RNA."""
    residue = make_residue3d("PSU", [], standard_residue_name="U")
    assert residue.molecule_type == Molecule.RNA


def test_molecule_type_modres_dna():
    """Modified residue with standard_residue_name resolved to DNA name should be DNA."""
    residue = make_residue3d("5MC", [], standard_residue_name="DC")
    assert residue.molecule_type == Molecule.DNA
