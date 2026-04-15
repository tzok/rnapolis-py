"""Shared test helpers and fixtures for rnapolis tests."""

import pandas as pd

from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.tertiary import Atom, Residue3D
from rnapolis.tertiary_v2 import ModifiedResidues, Residue as ResidueV2


def make_residue3d(
    auth_name: str, atom_names, standard_residue_name: str = ""
) -> Residue3D:
    """Build a minimal Residue3D for molecule_type unit tests."""
    label = ResidueLabel(chain="A", number=1, name=auth_name)
    auth = ResidueAuth(chain="A", number=1, icode=None, name=auth_name)
    atoms = tuple(
        Atom(
            entity_id=None,
            label=label,
            auth=auth,
            model=1,
            name=name,
            x=0.0,
            y=0.0,
            z=0.0,
            occupancy=1.0,
        )
        for name in atom_names
    )
    return Residue3D(
        label=label,
        auth=auth,
        model=1,
        one_letter_name=auth_name[0],
        atoms=atoms,
        standard_residue_name=standard_residue_name or auth_name,
    )


def build_pdb_residue_direct(residue_name: str, atom_names, modres=None) -> ResidueV2:
    """Build a v2 Residue directly (bypassing Structure filtering) for unit tests.

    Creates a minimal DataFrame with the given atom names and constructs
    a Residue object. If atom_names is empty, a single dummy atom is used
    so the DataFrame is never empty.

    Args:
        residue_name: Residue name (e.g. "A", "PSU").
        atom_names: Iterable of atom name strings.
        modres: Optional DataFrame with PDB-style MODRES columns.  Will be
            wrapped in :class:`ModifiedResidues` automatically.
    """
    if not atom_names:
        atom_names = ["_DUMMY"]
    records = []
    for serial, atom_name in enumerate(atom_names, 1):
        records.append(
            {
                "record_type": "ATOM",
                "serial": serial,
                "name": atom_name,
                "altLoc": None,
                "resName": residue_name,
                "chainID": "A",
                "resSeq": 1,
                "iCode": None,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "occupancy": 1.0,
                "tempFactor": 0.0,
                "element": "X" if atom_name == "_DUMMY" else atom_name[0],
                "charge": None,
                "model": 1,
            }
        )
    df = pd.DataFrame(records)
    df.attrs["format"] = "PDB"

    modres_obj = None
    if modres is not None:
        if not isinstance(modres, pd.DataFrame):
            modres_obj = modres
        else:
            modres.attrs.setdefault("format", "PDB")
            modres_obj = ModifiedResidues(modres)

    return ResidueV2(df, modres=modres_obj)
