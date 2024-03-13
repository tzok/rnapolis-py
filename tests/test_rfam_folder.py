import os

import pytest

from rnapolis.rfam_folder import generate_consensus_secondary_structure, parse_fasta

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_GK000002():
    fasta = parse_fasta("tests/GK000002.2-66269475-66272524.fa")
    generate_consensus_secondary_structure(fasta[0], "RF02540", False)
