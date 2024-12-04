import tempfile

from rnapolis.parser import parse_cif
from rnapolis.transformer import copy_from_to, replace_value


def test_replace_value():
    with open("tests/4gqj-assembly1.cif") as f:
        content = f.read()

    with tempfile.NamedTemporaryFile(mode="wt") as f:
        f.write(content)
        f.seek(0)
        org_atoms, _, _, _ = parse_cif(f)

    org_label_asym_id = set([atom.label.chain for atom in org_atoms if atom.label])
    org_auth_asym_id = set([atom.auth.chain for atom in org_atoms if atom.auth])
    assert org_label_asym_id == set(["A", "B", "A-2", "B-2"])
    assert org_auth_asym_id == set(["A", "B", "A-2", "B-2"])

    replaced_content, mapping = replace_value(
        content, "atom_site", "auth_asym_id", "ABCD"
    )
    assert mapping == {"A": "A", "B": "B", "A-2": "C", "B-2": "D"}

    with tempfile.NamedTemporaryFile(mode="rt+") as f:
        f.write(replaced_content)
        f.seek(0)
        rep_atoms, _, _, _ = parse_cif(f)

    rep_label_asym_id = set([atom.label.chain for atom in rep_atoms if atom.label])
    rep_auth_asym_id = set([atom.auth.chain for atom in rep_atoms if atom.auth])
    assert rep_label_asym_id == set(["A", "B", "A-2", "B-2"])
    assert rep_auth_asym_id == set(["A", "B", "C", "D"])


def test_copy_from_to():
    with open("tests/5it9.cif") as f:
        content = f.read()

    with tempfile.NamedTemporaryFile(mode="wt") as f:
        f.write(content)
        f.seek(0)
        org_atoms, _, _, _ = parse_cif(f)

    org_label_asym_id = set([atom.label.chain for atom in org_atoms if atom.label])
    org_auth_asym_id = set([atom.auth.chain for atom in org_atoms if atom.auth])
    assert org_label_asym_id == set(["HA", "IA"])
    assert org_auth_asym_id == set(["2", "i"])

    replaced_content = copy_from_to(
        content, "atom_site", "label_asym_id", "auth_asym_id"
    )

    with tempfile.NamedTemporaryFile(mode="rt+") as f:
        f.write(replaced_content)
        f.seek(0)
        rep_atoms, _, _, _ = parse_cif(f)

    rep_label_asym_id = set([atom.label.chain for atom in rep_atoms if atom.label])
    rep_auth_asym_id = set([atom.auth.chain for atom in rep_atoms if atom.auth])
    assert rep_label_asym_id == set(["HA", "IA"])
    assert rep_auth_asym_id == set(["HA", "IA"])
