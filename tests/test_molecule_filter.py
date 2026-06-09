import re
import sys
import tempfile

import pytest

from rnapolis.molecule_filter import filter_by_chains, filter_by_poly_types, main
from rnapolis.parser import parse_cif


def test_filter_by_poly_types():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    filtered = filter_by_poly_types(content, ["polyribonucleotide"], ["chem_comp"])
    assert re.search(r"^_entity\.id", filtered, re.MULTILINE) is not None
    assert re.search(r"^_chem_comp\.id", filtered, re.MULTILINE) is not None

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(filtered)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains == {"A", "B"}


def test_filter_by_chains():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    filtered = filter_by_chains(content, ["A", "C"], ["chem_comp"])
    assert re.search(r"^_entity\.id", filtered, re.MULTILINE) is not None
    assert re.search(r"^_chem_comp\.id", filtered, re.MULTILINE) is not None

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(filtered)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains == {"C", "D"}


def test_filter_by_chains_with_label_asym_id():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    filtered = filter_by_chains(
        content, ["A", "C"], ["chem_comp"], chain_id_source="label"
    )

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(filtered)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains == {"A", "C"}


def test_main_accepts_chain_id_source(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "molecule-filter",
            "--filter-by-chains",
            "A",
            "--chain-id-source",
            "label",
            "tests/1a9n.cif",
        ],
    )

    main()

    output = capsys.readouterr().out

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(output)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains == {"A"}


def test_filter_by_chains_rejects_unknown_chain_id_source():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    with pytest.raises(ValueError, match="chain_id_source"):
        filter_by_chains(content, ["A"], ["chem_comp"], chain_id_source="unknown")
