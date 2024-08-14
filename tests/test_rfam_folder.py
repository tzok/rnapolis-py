import os

import pytest
from rnapolis.rfam_folder import generate_consensus_secondary_structure, parse_fasta

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_GK000002():
    fasta = parse_fasta("tests/GK000002.2-66269475-66272524.fa")
    generate_consensus_secondary_structure(fasta[0], "RF02540", False)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_LJAJ01000009():
    fasta = parse_fasta("tests/LJAJ01000009.1-42413-42384.fa")
    result = generate_consensus_secondary_structure(fasta[0], "RF01315", False)
    sequence = result[0].split("\n")[1]
    assert len(sequence) == len(fasta[0].sequence)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_RF03339():
    fastas = parse_fasta("tests/RF03339.fa")
    for fasta in fastas:
        result = generate_consensus_secondary_structure(fasta, "RF03339", False)
        sequence = result[0].split("\n")[1]
        assert len(sequence) == len(fasta.sequence)
