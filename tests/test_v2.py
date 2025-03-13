import os

import pytest

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure


@pytest.fixture
def data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__))


def test_parse_4qln_formats(data_dir):
    """Test parsing 4qln in both PDB and mmCIF formats and compare residues."""
    # Load PDB and mmCIF files
    pdb_path = os.path.join(data_dir, "4qln.pdb")
    cif_path = os.path.join(data_dir, "4qln.cif")

    # Skip test if files don't exist
    if not (os.path.exists(pdb_path) and os.path.exists(cif_path)):
        pytest.skip(f"Test files not found: {pdb_path} or {cif_path}")

    # Parse both formats
    with open(pdb_path, "r") as pdb_file:
        pdb_atoms = parse_pdb_atoms(pdb_file)

    with open(cif_path, "r") as cif_file:
        cif_atoms = parse_cif_atoms(cif_file)

    # Create structures
    pdb_structure = Structure(pdb_atoms)
    cif_structure = Structure(cif_atoms)

    # Get residues
    pdb_residues = pdb_structure.residues
    cif_residues = cif_structure.residues

    # Basic checks
    assert len(pdb_residues) > 0, "No residues found in PDB file"
    assert len(cif_residues) > 0, "No residues found in mmCIF file"

    # Compare residue counts
    assert len(pdb_residues) == len(cif_residues), (
        f"Different number of residues: PDB={len(pdb_residues)}, mmCIF={len(cif_residues)}"
    )

    # Compare residue identifiers
    pdb_residue_ids = [
        (r.chain_id, r.residue_number, r.insertion_code) for r in pdb_residues
    ]
    cif_residue_ids = [
        (r.chain_id, r.residue_number, r.insertion_code) for r in cif_residues
    ]

    # Sort both lists to ensure consistent ordering
    pdb_residue_ids.sort()
    cif_residue_ids.sort()

    # Check if residue identifiers match
    for i, (pdb_id, cif_id) in enumerate(zip(pdb_residue_ids, cif_residue_ids)):
        assert pdb_id == cif_id, (
            f"Residue mismatch at position {i}: PDB={pdb_id}, mmCIF={cif_id}"
        )

    # Create a mapping from residue ID to residue name for both formats
    pdb_id_to_name = {
        (r.chain_id, r.residue_number, r.insertion_code): r.residue_name
        for r in pdb_residues
    }
    cif_id_to_name = {
        (r.chain_id, r.residue_number, r.insertion_code): r.residue_name
        for r in cif_residues
    }

    # Check if residue names match for each residue ID
    for res_id in pdb_id_to_name:
        assert res_id in cif_id_to_name, f"Residue ID {res_id} not found in mmCIF"
        assert pdb_id_to_name[res_id] == cif_id_to_name[res_id], (
            f"Residue name mismatch for {res_id}: PDB={pdb_id_to_name[res_id]}, mmCIF={cif_id_to_name[res_id]}"
        )
