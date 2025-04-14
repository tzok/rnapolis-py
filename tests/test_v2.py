import io
import os

import numpy as np
import pandas as pd
import pytest

from rnapolis.parser_v2 import (
    fit_to_pdb,
    parse_cif_atoms,
    parse_pdb_atoms,
    write_cif,
    write_pdb,
)
from rnapolis.tertiary_v2 import Structure, calculate_torsion_angle

# Define column lists for comparison helper
PDB_ESSENTIAL_COLS = [
    "record_type",
    "serial",
    "name",
    "altLoc",
    "resName",
    "chainID",
    "resSeq",
    "iCode",
    "x",
    "y",
    "z",
    "occupancy",
    "tempFactor",
    "element",
    "charge",
    "model",
]
PDB_NUMERIC_COLS = [
    "serial",
    "resSeq",
    "x",
    "y",
    "z",
    "occupancy",
    "tempFactor",
    "model",
]
PDB_OBJECT_COLS = [col for col in PDB_ESSENTIAL_COLS if col not in PDB_NUMERIC_COLS]
PDB_SORT_COLS = ["model", "chainID", "resSeq", "serial"]


CIF_ESSENTIAL_COLS = [
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
]
# Note: Some CIF numeric cols might be Int64 due to parsing Nones
CIF_NUMERIC_COLS = [
    "id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "pdbx_PDB_model_num",
]
CIF_OBJECT_COLS = [col for col in CIF_ESSENTIAL_COLS if col not in CIF_NUMERIC_COLS]
CIF_SORT_COLS = ["pdbx_PDB_model_num", "auth_asym_id", "auth_seq_id", "id"]


def compare_atom_dfs(
    df1, df2, essential_cols, numeric_cols, object_cols, sort_cols, rtol=1e-5, atol=1e-5
):
    """Compares two atom DataFrames for essential column equality."""
    # Select only essential columns that exist in both dataframes
    cols1 = [col for col in essential_cols if col in df1.columns]
    cols2 = [col for col in essential_cols if col in df2.columns]
    common_cols = list(set(cols1) & set(cols2))

    # Check if essential sorting columns are present
    sort_cols_present1 = [col for col in sort_cols if col in df1.columns]
    sort_cols_present2 = [col for col in sort_cols if col in df2.columns]
    if not sort_cols_present1 or not sort_cols_present2 or sort_cols_present1 != sort_cols_present2:
         raise ValueError(f"Cannot compare DataFrames: Missing or mismatched sort columns. DF1: {sort_cols_present1}, DF2: {sort_cols_present2}")

    df1_comp = df1[common_cols].copy()
    df2_comp = df2[common_cols].copy()

    # Ensure consistent sorting
    df1_comp = df1_comp.sort_values(by=sort_cols_present1).reset_index(drop=True)
    df2_comp = df2_comp.sort_values(by=sort_cols_present2).reset_index(drop=True)

    # Compare object/categorical columns (handle None/NaN consistently)
    current_object_cols = [col for col in object_cols if col in common_cols]
    for col in current_object_cols:
        s1 = df1_comp[col].fillna("").astype(str)
        s2 = df2_comp[col].fillna("").astype(str)
        pd.testing.assert_series_equal(
            s1, s2, check_names=False, check_dtype=False, obj=f"{col} column comparison"
        )

    # Compare numeric columns with tolerance
    current_numeric_cols = [col for col in numeric_cols if col in common_cols]
    for col in current_numeric_cols:
        # Convert both to float64 for comparison, coercing errors
        v1 = pd.to_numeric(df1_comp[col], errors="coerce").astype(np.float64)
        v2 = pd.to_numeric(df2_comp[col], errors="coerce").astype(np.float64)

        # Check NaNs match after coercion
        nan_mask1 = v1.isna()
        nan_mask2 = v2.isna()
        assert nan_mask1.equals(nan_mask2), f"NaN pattern mismatch in {col}"

        # Compare non-NaN values
        mask = ~nan_mask1
        if mask.any():
             np.testing.assert_allclose(
                 v1[mask].values,
                 v2[mask].values,
                 rtol=rtol,
                 atol=atol,
                 err_msg=f"{col} column comparison failed",
             )


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


def test_pdb_cif_pdb_roundtrip(data_dir):
    """Test PDB -> CIF -> PDB conversion preserves essential data."""
    pdb_path = os.path.join(data_dir, "1ATO.pdb")
    if not os.path.exists(pdb_path):
        pytest.skip(f"Test file not found: {pdb_path}")

    # 1. Parse Original PDB
    with open(pdb_path, "r") as f:
        df_orig_pdb = parse_pdb_atoms(f)
    assert not df_orig_pdb.empty, "Original PDB parsing failed"

    # 2. Write CIF
    cif_str = write_cif(df_orig_pdb)
    assert cif_str is not None and "atom_site" in cif_str, "Writing CIF failed"

    # 3. Parse Intermediate CIF
    df_intermediate_cif = parse_cif_atoms(io.StringIO(cif_str))
    assert not df_intermediate_cif.empty, "Parsing intermediate CIF failed"

    # 4. Write Final PDB
    pdb_final_str = write_pdb(df_intermediate_cif)
    assert pdb_final_str is not None and "ATOM" in pdb_final_str, "Writing final PDB failed"

    # 5. Parse Final PDB
    df_final_pdb = parse_pdb_atoms(io.StringIO(pdb_final_str))
    assert not df_final_pdb.empty, "Parsing final PDB failed"

    # 6. Compare Original PDB with Final PDB
    compare_atom_dfs(
        df_orig_pdb,
        df_final_pdb,
        PDB_ESSENTIAL_COLS,
        PDB_NUMERIC_COLS,
        PDB_OBJECT_COLS,
        PDB_SORT_COLS,
    )


def test_cif_pdb_cif_roundtrip(data_dir):
    """Test CIF -> PDB -> CIF conversion preserves essential data."""
    cif_path = os.path.join(data_dir, "1ehz-assembly-1.cif")
    if not os.path.exists(cif_path):
        pytest.skip(f"Test file not found: {cif_path}")

    # 1. Parse Original CIF
    with open(cif_path, "r") as f:
        df_orig_cif = parse_cif_atoms(f)
    assert not df_orig_cif.empty, "Original CIF parsing failed"

    # 2. Fit to PDB (if necessary) and Write PDB
    try:
        # fit_to_pdb returns original df if it fits, otherwise a modified copy
        df_to_write_pdb = fit_to_pdb(df_orig_cif)
        pdb_str = write_pdb(df_to_write_pdb)
        assert pdb_str is not None and "ATOM" in pdb_str, "Writing PDB failed"
    except ValueError as e:
        pytest.skip(f"Cannot fit {cif_path} to PDB, skipping roundtrip test: {e}")

    # 3. Parse Intermediate PDB
    df_from_pdb = parse_pdb_atoms(io.StringIO(pdb_str))
    assert not df_from_pdb.empty, "Parsing intermediate PDB failed"

    # 4. Write Final CIF
    cif_final_str = write_cif(df_from_pdb)
    assert cif_final_str is not None and "atom_site" in cif_final_str, "Writing final CIF failed"

    # 5. Parse Final CIF
    df_final_cif = parse_cif_atoms(io.StringIO(cif_final_str))
    assert not df_final_cif.empty, "Parsing final CIF failed"

    # 6. Compare Original CIF with Final CIF
    compare_atom_dfs(
        df_orig_cif,
        df_final_cif,
        CIF_ESSENTIAL_COLS,
        CIF_NUMERIC_COLS,
        CIF_OBJECT_COLS,
        CIF_SORT_COLS,
    )
