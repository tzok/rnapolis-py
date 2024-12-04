import tempfile

from rnapolis.molecule_filter import filter_by_chains, filter_by_poly_types
from rnapolis.parser import parse_cif


def test_filter_by_poly_types():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    filtered = filter_by_poly_types(content, ["polyribonucleotide"])

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(filtered)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains == {"A", "B"}


def test_filter_by_chains():
    with open("tests/1a9n.cif") as f:
        content = f.read()

    filtered = filter_by_chains(content, ["A", "C"])

    with tempfile.NamedTemporaryFile("rt+") as f:
        f.write(filtered)
        f.seek(0)
        atoms, _, _, _ = parse_cif(f)

    chains = set([atom.label.chain for atom in atoms if atom.label])
    assert chains >= {"A", "C"}
