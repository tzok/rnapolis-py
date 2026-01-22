from rnapolis.adapters.barnaba import parse_barnaba_output
from rnapolis.common import LeontisWesthof, ResidueAuth, ResidueLabel
from rnapolis.tertiary import Atom, Residue3D, Structure3D


def _make_residue3d(chain: str, number: int, auth_name: str, one_letter: str) -> Residue3D:
    label = ResidueLabel(chain=chain, number=number, name=auth_name)
    auth = ResidueAuth(chain=chain, number=number, icode=None, name=auth_name)

    atom_names = (
        set(Residue3D.phosphate_atoms)
        | set(Residue3D.sugar_atoms)
        | set(Residue3D.nucleobase_heavy_atoms[one_letter])
    )

    # Keep the key covalent distances small so Residue3D.is_nucleotide == True.
    coords = {
        "P": (0.0, 0.0, 0.0),
        "O5'": (1.0, 0.0, 0.0),
        "C1'": (0.0, 1.0, 0.0),
        # Purines use N9 for glycosidic bond detection; pyrimidines use N1.
        "N9": (0.0, 1.5, 0.0),
        "N1": (0.0, 1.5, 0.0),
    }

    atoms = tuple(
        Atom(
            entity_id=None,
            label=label,
            auth=auth,
            model=1,
            name=name,
            x=coords.get(name, (0.0, 0.0, 0.0))[0],
            y=coords.get(name, (0.0, 0.0, 0.0))[1],
            z=coords.get(name, (0.0, 0.0, 0.0))[2],
            occupancy=1.0,
        )
        for name in sorted(atom_names)
    )
    return Residue3D(label=label, auth=auth, model=1, one_letter_name=one_letter, atoms=atoms)


def test_barnaba_sequence_negative_residue_number(tmp_path):
    pairing_path = tmp_path / "example_pairing.out"
    pairing_path.write_text(
        "\n".join(
            [
                "# sequence DA_-1_0-G_1_0",
                "DA_-1_0 G_1_0 WCc",
                "",
            ]
        )
    )

    structure3d = Structure3D(
        [
            _make_residue3d(chain="A", number=-1, auth_name="DA", one_letter="A"),
            _make_residue3d(chain="A", number=1, auth_name="G", one_letter="G"),
        ]
    )

    base_interactions = parse_barnaba_output([str(pairing_path)], structure3d)

    assert len(base_interactions.base_pairs) == 1
    bp = base_interactions.base_pairs[0]
    assert {bp.nt1.auth.number, bp.nt2.auth.number} == {-1, 1}
    assert {bp.nt1.auth.name, bp.nt2.auth.name} == {"DA", "G"}
    assert bp.lw == LeontisWesthof.cWW
