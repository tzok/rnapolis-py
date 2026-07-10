"""Tests for rnapolis.chain_groups."""

import os

import pytest

from rnapolis.chain_groups import find_na_chain_groups

# Minimal PDB with two nucleotide chains far apart (100 A).
# Each chain has a single adenosine with C1', so no contacts at 15 A.
# Coordinates are spread far enough that no pair is within threshold.
_TWO_CHAIN_FAR_PDB = """\
ATOM      1  P     A A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  O5'   A A   1       1.500   0.000   0.000  1.00  0.00           O
ATOM      3  C5'   A A   1       2.500   0.000   0.000  1.00  0.00           C
ATOM      4  C4'   A A   1       3.000   1.000   0.000  1.00  0.00           C
ATOM      5  O4'   A A   1       4.000   0.800   0.000  1.00  0.00           O
ATOM      6  C3'   A A   1       3.500   2.000   0.000  1.00  0.00           C
ATOM      7  O3'   A A   1       4.500   2.500   0.000  1.00  0.00           O
ATOM      8  C2'   A A   1       2.800   2.500   1.000  1.00  0.00           C
ATOM      9  O2'   A A   1       2.500   3.500   1.000  1.00  0.00           O
ATOM     10  C1'   A A   1       4.200   1.800   0.500  1.00  0.00           C
ATOM     11  N9    A A   1       5.200   2.500   0.500  1.00  0.00           N
ATOM     12  C8    A A   1       6.300   2.300   0.500  1.00  0.00           C
ATOM     13  N7    A A   1       6.500   1.200   0.500  1.00  0.00           N
ATOM     14  C5    A A   1       5.400   1.100   0.500  1.00  0.00           C
ATOM     15  C6    A A   1       5.200  -0.200   0.500  1.00  0.00           C
ATOM     16  N6    A A   1       4.200  -0.800   0.500  1.00  0.00           N
ATOM     17  N1    A A   1       6.300  -0.800   0.500  1.00  0.00           N
ATOM     18  C2    A A   1       7.200  -0.200   0.500  1.00  0.00           C
ATOM     19  N3    A A   1       7.000   1.100   0.500  1.00  0.00           N
ATOM     20  C4    A A   1       6.000   2.000   0.500  1.00  0.00           C
ATOM     21  P     G B   1     100.000   0.000   0.000  1.00  0.00           P
ATOM     22  O5'   G B   1     101.500   0.000   0.000  1.00  0.00           O
ATOM     23  C5'   G B   1     102.500   0.000   0.000  1.00  0.00           C
ATOM     24  C4'   G B   1     103.000   1.000   0.000  1.00  0.00           C
ATOM     25  O4'   G B   1     104.000   0.800   0.000  1.00  0.00           O
ATOM     26  C3'   G B   1     103.500   2.000   0.000  1.00  0.00           C
ATOM     27  O3'   G B   1     104.500   2.500   0.000  1.00  0.00           O
ATOM     28  C2'   G B   1     102.800   2.500   1.000  1.00  0.00           C
ATOM     29  O2'   G B   1     102.500   3.500   1.000  1.00  0.00           O
ATOM     30  C1'   G B   1     104.200   1.800   0.500  1.00  0.00           C
ATOM     31  N9    G B   1     105.200   2.500   0.500  1.00  0.00           N
ATOM     32  C8    G B   1     106.300   2.300   0.500  1.00  0.00           C
ATOM     33  N7    G B   1     106.500   1.200   0.500  1.00  0.00           N
ATOM     34  C5    G B   1     105.400   1.100   0.500  1.00  0.00           C
ATOM     35  C6    G B   1     105.200  -0.200   0.500  1.00  0.00           C
ATOM     36  O6    G B   1     104.200  -0.800   0.500  1.00  0.00           O
ATOM     37  N1    G B   1     106.300  -0.800   0.500  1.00  0.00           N
ATOM     38  C2    G B   1     107.200  -0.200   0.500  1.00  0.00           C
ATOM     39  N3    G B   1     107.000   1.100   0.500  1.00  0.00           N
ATOM     40  C4    G B   1     106.000   2.000   0.500  1.00  0.00           C
END
"""

# Two chains close together -- C1' atoms within 10 A of each other.
_TWO_CHAIN_CLOSE_PDB = """\
ATOM      1  C1'   A A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N9    A A   1       1.500   0.000   0.000  1.00  0.00           N
ATOM      3  C4    A A   1       2.500   0.500   0.000  1.00  0.00           C
ATOM      4  C5    A A   1       2.000   1.500   0.000  1.00  0.00           C
ATOM      5  N7    A A   1       2.800   2.500   0.000  1.00  0.00           N
ATOM      6  C8    A A   1       1.800   2.200   0.000  1.00  0.00           C
ATOM      7  N1    A A   1       3.500  -0.500   0.000  1.00  0.00           N
ATOM      8  C2    A A   1       4.500  -0.200   0.000  1.00  0.00           C
ATOM      9  N3    A A   1       4.200   1.100   0.000  1.00  0.00           N
ATOM     10  C6    A A   1       3.200   1.200   0.000  1.00  0.00           C
ATOM     11  N6    A A   1       3.800   2.200   0.000  1.00  0.00           N
ATOM     12  C1'   G B   1       8.000   0.000   0.000  1.00  0.00           C
ATOM     13  N9    G B   1       9.500   0.000   0.000  1.00  0.00           N
ATOM     14  C4    G B   1      10.500   0.500   0.000  1.00  0.00           C
ATOM     15  C5    G B   1      10.000   1.500   0.000  1.00  0.00           C
ATOM     16  N7    G B   1      10.800   2.500   0.000  1.00  0.00           N
ATOM     17  C8    G B   1       9.800   2.200   0.000  1.00  0.00           C
ATOM     18  N1    G B   1      11.500  -0.500   0.000  1.00  0.00           N
ATOM     19  C2    G B   1      12.500  -0.200   0.000  1.00  0.00           C
ATOM     20  N3    G B   1      12.200   1.100   0.000  1.00  0.00           N
ATOM     21  C4    G B   1      11.000   1.100   0.000  1.00  0.00           C
ATOM     22  C6    G B   1      11.200   1.200   0.000  1.00  0.00           C
ATOM     23  O6    G B   1      11.800   2.200   0.000  1.00  0.00           O
END
"""


@pytest.fixture
def data_dir():
    return os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Unit tests -- synthetic PDB data
# ---------------------------------------------------------------------------


def test_two_chains_far_apart():
    """Two chains with C1' atoms 100 A apart should produce two singleton groups."""
    groups = find_na_chain_groups(_TWO_CHAIN_FAR_PDB)
    assert len(groups) == 2
    assert {"A"} in groups
    assert {"B"} in groups


def test_two_chains_close():
    """Two chains with C1' atoms ~8 A apart should produce one merged group."""
    groups = find_na_chain_groups(_TWO_CHAIN_CLOSE_PDB)
    assert len(groups) == 1
    assert groups[0] == {"A", "B"}


def test_large_threshold_merges_distant_chains():
    """With a very large threshold, even far-apart chains should merge."""
    groups = find_na_chain_groups(_TWO_CHAIN_FAR_PDB, distance_threshold=200.0)
    assert len(groups) == 1
    assert groups[0] == {"A", "B"}


def test_custom_atom_name_c5_prime():
    """Using C5' instead of C1' should still find the same chains."""
    groups = find_na_chain_groups(_TWO_CHAIN_FAR_PDB, atom_name="C5'")
    assert len(groups) == 2
    assert {"A"} in groups
    assert {"B"} in groups


def test_empty_content():
    """An empty string should return an empty list."""
    assert find_na_chain_groups("") == []


def test_no_c1_atoms():
    """Content with no C1' atoms should return an empty list."""
    pdb_no_c1 = """\
ATOM      1  CA    ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N     ALA A   1       1.500   0.000   0.000  1.00  0.00           N
END
"""
    assert find_na_chain_groups(pdb_no_c1) == []


# A protein chain (E) with a bound ATP cofactor.  ATP has a ribose with
# C1', but its residue name is "ATP" -- not a canonical NA name.  The NA
# chain (A) is a normal adenosine.  Chain E must be excluded.
# Note: PDB columns are fixed-width — 3-char residue names (ALA, ATP)
# have different spacing than 1-char names (A).
_PROTEIN_WITH_COFACTOR_PDB = """\
ATOM      1  C1'   A A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N9    A A   1       1.500   0.000   0.000  1.00  0.00           N
ATOM      3  N    ALA E   1      50.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA   ALA E   1      51.000   0.000   0.000  1.00  0.00           C
ATOM      5  C    ALA E   1      52.000   0.000   0.000  1.00  0.00           C
ATOM      6  O    ALA E   1      53.000   0.000   0.000  1.00  0.00           O
HETATM    7  C1'  ATP E   2      55.000   0.000   0.000  1.00  0.00           C
HETATM    8  N9   ATP E   2      56.500   0.000   0.000  1.00  0.00           N
HETATM    9  PA   ATP E   2      57.000   1.000   0.000  1.00  0.00           P
END
"""


def test_protein_chain_with_cofactor_excluded():
    """A protein chain with a bound ATP (which has C1') must not appear in results."""
    groups = find_na_chain_groups(_PROTEIN_WITH_COFACTOR_PDB)
    assert len(groups) == 1
    assert groups[0] == {"A"}
    all_chains = set()
    for g in groups:
        all_chains.update(g)
    assert "E" not in all_chains


def test_single_chain():
    """A single nucleic-acid chain should produce one singleton group."""
    single = _TWO_CHAIN_FAR_PDB.replace("B", "A").replace("G A", "G A")
    groups = find_na_chain_groups(single)
    assert len(groups) == 1
    assert groups[0] == {"A"}


# ---------------------------------------------------------------------------
# Integration tests -- real test data
# ---------------------------------------------------------------------------


def test_1a4d_two_interacting_chains(data_dir):
    """1A4D has two interacting nucleic-acid chains A and B."""
    cif_path = os.path.join(data_dir, "1A4D_1_A-B.cif")
    if not os.path.exists(cif_path):
        pytest.skip(f"Test file not found: {cif_path}")

    with open(cif_path) as f:
        content = f.read()

    groups = find_na_chain_groups(content)
    assert len(groups) == 1
    assert groups[0] == {"A", "B"}


def test_1ehz_single_chain(data_dir):
    """1ehz is a single-chain tRNA structure."""
    cif_path = os.path.join(data_dir, "1ehz-assembly-1.cif")
    if not os.path.exists(cif_path):
        pytest.skip(f"Test file not found: {cif_path}")

    with open(cif_path) as f:
        content = f.read()

    groups = find_na_chain_groups(content)
    assert len(groups) == 1
    assert groups[0] == {"A"}


def test_4qln_single_chain_pdb(data_dir):
    """4qln PDB has a single nucleic-acid chain A."""
    pdb_path = os.path.join(data_dir, "4qln.pdb")
    if not os.path.exists(pdb_path):
        pytest.skip(f"Test file not found: {pdb_path}")

    with open(pdb_path) as f:
        content = f.read()

    groups = find_na_chain_groups(content)
    assert len(groups) == 1
    assert groups[0] == {"A"}


def test_6inq_two_chains(data_dir):
    """6INQ has chains N and T; check if they are grouped correctly."""
    cif_path = os.path.join(data_dir, "6INQ.cif")
    if not os.path.exists(cif_path):
        pytest.skip(f"Test file not found: {cif_path}")

    with open(cif_path) as f:
        content = f.read()

    groups = find_na_chain_groups(content)
    chain_set = set()
    for g in groups:
        chain_set.update(g)
    assert chain_set == {"N", "T"}
