import os

import numpy as np
import pandas as pd
import pytest

from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure


@pytest.fixture
def data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__))


def test_parse_4qln_formats(data_dir):
    """Test parsing 4qln in both PDB and mmCIF formats and compare residues and torsion angles."""
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

    # Calculate torsion angles for both structures
    pdb_torsion_df = pdb_structure.torsion_angles
    cif_torsion_df = cif_structure.torsion_angles

    # Check if torsion angle DataFrames have the same shape
    assert pdb_torsion_df.shape == cif_torsion_df.shape, (
        f"Different torsion angle DataFrame shapes: PDB={pdb_torsion_df.shape}, mmCIF={cif_torsion_df.shape}"
    )

    # Sort both DataFrames by chain_id, residue_number, and insertion_code for consistent comparison
    pdb_torsion_df = pdb_torsion_df.sort_values(
        by=["chain_id", "residue_number", "insertion_code"]
    ).reset_index(drop=True)

    cif_torsion_df = cif_torsion_df.sort_values(
        by=["chain_id", "residue_number", "insertion_code"]
    ).reset_index(drop=True)

    # Compare residue identifiers in torsion angle DataFrames
    pd.testing.assert_series_equal(
        pdb_torsion_df["chain_id"],
        cif_torsion_df["chain_id"],
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        pdb_torsion_df["residue_number"],
        cif_torsion_df["residue_number"],
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        pdb_torsion_df["residue_name"],
        cif_torsion_df["residue_name"],
        check_names=False,
        check_dtype=False,
    )

    # Compare torsion angle values with a tolerance
    angle_columns = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
    for col in angle_columns:
        # Skip columns that might not exist in both DataFrames
        if col not in pdb_torsion_df.columns or col not in cif_torsion_df.columns:
            continue

        # Get non-NaN values that exist in both DataFrames
        pdb_values = pdb_torsion_df[col]
        cif_values = cif_torsion_df[col]

        # Check if the same values are NaN in both DataFrames
        assert pdb_values.isna().equals(cif_values.isna()), (
            f"Different NaN patterns in {col} angle"
        )

        # Compare non-NaN values with tolerance
        mask = ~pdb_values.isna()
        if mask.any():
            pdb_non_nan = pdb_values[mask].values
            cif_non_nan = cif_values[mask].values

            # Allow a small tolerance for floating-point differences
            np.testing.assert_allclose(
                pdb_non_nan,
                cif_non_nan,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Torsion angle values for {col} don't match between PDB and mmCIF",
            )


def test_torsion_angle_calculation():
    """Test the torsion angle calculation function."""
    # Define four points that form a known torsion angle
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 0.0, 0.0])
    a3 = np.array([0.0, 1.0, 0.0])
    a4 = np.array([0.0, 1.0, 1.0])

    # Calculate the torsion angle
    from rnapolis.tertiary_v2 import calculate_torsion_angle

    angle = calculate_torsion_angle(a1, a2, a3, a4)

    # The expected angle is pi/2 radians (90 degrees)
    assert abs(angle - np.pi / 2) < 1e-6, (
        f"Expected angle close to pi/2 radians, got {angle}"
    )

    # Test with collinear points
    a1 = np.array([0.0, 0.0, 0.0])
    a2 = np.array([1.0, 0.0, 0.0])
    a3 = np.array([2.0, 0.0, 0.0])
    a4 = np.array([3.0, 0.0, 0.0])

    angle = calculate_torsion_angle(a1, a2, a3, a4)
    assert np.isnan(angle), f"Expected NaN for collinear points, got {angle}"


def test_connected_residues_and_torsion_angles(data_dir):
    """Test finding connected residues and calculating torsion angles."""
    # Load PDB file
    pdb_path = os.path.join(data_dir, "4qln.pdb")

    # Skip test if file doesn't exist
    if not os.path.exists(pdb_path):
        pytest.skip(f"Test file not found: {pdb_path}")

    # Parse PDB file
    with open(pdb_path, "r") as pdb_file:
        pdb_atoms = parse_pdb_atoms(pdb_file)

    # Create structure
    structure = Structure(pdb_atoms)

    # Find connected residues
    segments = structure.connected_residues

    # Check that we found at least one segment
    assert len(segments) > 0, "No connected residue segments found"

    # Check that each segment has at least 2 residues
    for segment in segments:
        assert len(segment) >= 2, f"Segment has fewer than 2 residues: {segment}"

    # Calculate torsion angles
    torsion_df = structure.torsion_angles

    # Check that the DataFrame has the expected columns
    expected_columns = [
        "chain_id",
        "residue_number",
        "insertion_code",
        "residue_name",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "chi",
    ]
    for col in expected_columns:
        assert col in torsion_df.columns, (
            f"Expected column {col} not found in torsion angles DataFrame"
        )

    # Check that we have some torsion angle values
    assert len(torsion_df) > 0, "No torsion angles calculated"

    # Check that at least some angles are not null
    for angle in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]:
        assert torsion_df[angle].notna().any(), f"No valid {angle} angles calculated"
