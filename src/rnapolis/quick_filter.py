#!/usr/bin/env python3
import argparse
import io
import os
import re
import tempfile
from typing import Iterable, Optional, Set

import pandas as pd
from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory

from rnapolis.parser import is_cif
from rnapolis.parser_v2 import parse_cif_atoms, parse_pdb_atoms
from rnapolis.tertiary_v2 import Structure
from rnapolis.util import handle_input_file

WATER_RESIDUE_NAMES = {
    "HOH",
    "WAT",
    "H2O",
    "DOD",
}
ION_RESIDUE_NAMES = {
    "AG",
    "AL",
    "AU",
    "BA",
    "BR",
    "CA",
    "CD",
    "CL",
    "CO",
    "CS",
    "CU",
    "F",
    "FE",
    "HG",
    "I",
    "IOD",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NI",
    "PB",
    "RB",
    "SR",
    "ZN",
}
PRIMARY_ALTLOC_VALUES = {"", ".", "?", "A"}


def _normalize_chain_ids(chains: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if chains is None:
        return None
    chain_list = [chain.strip() for chain in chains if chain.strip()]
    return set(chain_list) if chain_list else None


def _chain_column(atoms: pd.DataFrame) -> Optional[str]:
    if atoms.attrs.get("format") == "PDB":
        return "chainID"
    if "auth_asym_id" in atoms.columns:
        return "auth_asym_id"
    if "label_asym_id" in atoms.columns:
        return "label_asym_id"
    return None


def _altloc_column(atoms: pd.DataFrame) -> Optional[str]:
    if atoms.attrs.get("format") == "PDB":
        return "altLoc"
    if "label_alt_id" in atoms.columns:
        return "label_alt_id"
    if "pdbx_auth_alt_id" in atoms.columns:
        return "pdbx_auth_alt_id"
    return None


def _model_column(atoms: pd.DataFrame) -> Optional[str]:
    if atoms.attrs.get("format") == "PDB":
        return "model"
    if "pdbx_PDB_model_num" in atoms.columns:
        return "pdbx_PDB_model_num"
    return None


def _residue_name(residue) -> str:
    return residue.residue_name.strip().upper()


def _is_water(residue_name: str) -> bool:
    return residue_name in WATER_RESIDUE_NAMES


def _is_ion(residue_name: str) -> bool:
    return residue_name in ION_RESIDUE_NAMES


def filter_atoms_df(
    atoms: pd.DataFrame,
    mode: str,
    keep_ligands: bool = False,
    keep_waters: bool = False,
    keep_ions: bool = False,
    keep_altlocs: bool = False,
    chains: Optional[Iterable[str]] = None,
    model: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter atoms for the selected mode and optional inclusions.

    Parameters:
    -----------
    atoms : pd.DataFrame
        Atom DataFrame from parser_v2.
    mode : str
        "nucleic-acid" or "protein".
    keep_ligands : bool
        Keep non-water, non-ion ligands.
    keep_waters : bool
        Keep water residues.
    keep_ions : bool
        Keep ionic residues.
    keep_altlocs : bool
        Keep non-primary alternate locations.
    chains : Optional[Iterable[str]]
        Chain IDs to keep.
    model : Optional[int]
        Model number to keep.
    """
    if atoms.empty:
        return atoms

    if mode not in {"nucleic-acid", "protein"}:
        raise ValueError("mode must be 'nucleic-acid' or 'protein'")

    chains_set = _normalize_chain_ids(chains)
    structure = Structure(atoms)
    keep_indices = set()

    for residue in structure.residues:
        residue_name = _residue_name(residue)
        is_primary = (
            residue.is_nucleotide if mode == "nucleic-acid" else residue.is_amino_acid
        )
        if is_primary:
            keep_indices.update(residue.atoms.index)
            continue

        if _is_water(residue_name):
            if keep_waters:
                keep_indices.update(residue.atoms.index)
            continue

        if _is_ion(residue_name):
            if keep_ions:
                keep_indices.update(residue.atoms.index)
            continue

        if keep_ligands and not residue.is_nucleotide and not residue.is_amino_acid:
            keep_indices.update(residue.atoms.index)

    if keep_indices:
        filtered = atoms.loc[sorted(keep_indices)].copy()
    else:
        filtered = atoms.iloc[0:0].copy()

    chain_col = _chain_column(filtered)
    if chains_set and chain_col and chain_col in filtered.columns:
        chain_series = filtered[chain_col].astype(str)
        filtered = filtered[chain_series.isin(chains_set)].copy()

    model_col = _model_column(filtered)
    if model is not None and model_col and model_col in filtered.columns:
        filtered = filtered[filtered[model_col] == model].copy()

    altloc_col = _altloc_column(filtered)
    if not keep_altlocs and altloc_col and altloc_col in filtered.columns:
        altloc_series = filtered[altloc_col]
        altloc_str = altloc_series.astype(str)
        keep_mask = altloc_series.isna() | altloc_str.isin(PRIMARY_ALTLOC_VALUES)
        filtered = filtered[keep_mask].copy()

    filtered.attrs["format"] = atoms.attrs.get("format")
    return filtered


def _format_cif_value(value) -> str:
    if pd.isna(value):
        return "?"
    return str(value)


def _normalize_cif_content(content: str) -> str:
    content = re.sub(r"(^\s*data_)\s*$", r"\1unnamed", content, flags=re.MULTILINE)
    for axis in ("x", "y", "z"):
        content = re.sub(
            rf"(_atom_site\.)cartn_{axis}\b",
            rf"\1Cartn_{axis}",
            content,
            flags=re.IGNORECASE,
        )
    return content


def _read_cif_data(content: str):
    adapter = IoAdapterPy()
    content = _normalize_cif_content(content)
    with tempfile.NamedTemporaryFile(mode="wt", suffix=".cif", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return adapter.readFile(tmp_path)
    finally:
        os.remove(tmp_path)


def _filter_cif_content(content: str, filtered: pd.DataFrame) -> str:
    data = _read_cif_data(content)
    if len(data) == 0 or "atom_site" not in data[0].getObjNameList():
        return content

    category_obj = data[0].getObj("atom_site")
    attributes = category_obj.getAttributeList()
    rows = []

    if len(filtered) > 0:
        for _, row in filtered.iterrows():
            row_values = []
            for attribute in attributes:
                if attribute in filtered.columns:
                    value = row[attribute]
                else:
                    value = "?"
                row_values.append(_format_cif_value(value))
            rows.append(row_values)

    data[0].replace(DataCategory("atom_site", attributes, rows))

    adapter = IoAdapterPy()
    with tempfile.NamedTemporaryFile(mode="rt+", suffix=".cif") as tmp:
        adapter.writeFile(tmp.name, data)
        tmp.seek(0)
        return tmp.read()


def _filter_pdb_content(content: str, filtered: pd.DataFrame) -> str:
    serial_pairs = set()
    if len(filtered) > 0:
        if "serial" in filtered.columns and "model" in filtered.columns:
            numeric_values = filtered[["serial", "model"]].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric_values = numeric_values.dropna()
            for serial_value, model_value in numeric_values.itertuples(
                index=False, name=None
            ):
                serial_pairs.add((int(model_value), int(serial_value)))

    output_lines = []
    current_model = 1
    for line in content.splitlines(keepends=True):
        record_type = line[:6].strip()

        if record_type == "MODEL":
            try:
                current_model = int(line[10:14].strip())
            except ValueError:
                current_model = 1
            output_lines.append(line)
            continue

        if record_type in {"ATOM", "HETATM"}:
            try:
                serial = int(line[6:11].strip())
            except ValueError:
                continue
            if (current_model, serial) in serial_pairs:
                output_lines.append(line)
            continue

        output_lines.append(line)

    return "".join(output_lines)


def filter_content(
    content: str,
    mode: str,
    keep_ligands: bool = False,
    keep_waters: bool = False,
    keep_ions: bool = False,
    keep_altlocs: bool = False,
    chains: Optional[Iterable[str]] = None,
    model: Optional[int] = None,
) -> str:
    """Filter PDB or mmCIF content and return the filtered file content."""
    format_is_cif = is_cif(io.StringIO(content))
    if format_is_cif:
        content = _normalize_cif_content(content)
        atoms = parse_cif_atoms(content)
    else:
        atoms = parse_pdb_atoms(content)

    filtered = filter_atoms_df(
        atoms,
        mode=mode,
        keep_ligands=keep_ligands,
        keep_waters=keep_waters,
        keep_ions=keep_ions,
        keep_altlocs=keep_altlocs,
        chains=chains,
        model=model,
    )

    if format_is_cif:
        return _filter_cif_content(content, filtered)

    return _filter_pdb_content(content, filtered)


def filter_file(
    path: str,
    mode: str,
    keep_ligands: bool = False,
    keep_waters: bool = False,
    keep_ions: bool = False,
    keep_altlocs: bool = False,
    chains: Optional[Iterable[str]] = None,
    model: Optional[int] = None,
) -> str:
    file_handle = handle_input_file(path)
    content = file_handle.read()
    file_handle.close()
    return filter_content(
        content,
        mode=mode,
        keep_ligands=keep_ligands,
        keep_waters=keep_waters,
        keep_ions=keep_ions,
        keep_altlocs=keep_altlocs,
        chains=chains,
        model=model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter PDB or mmCIF atoms while preserving non-atom records."
    )
    parser.add_argument(
        "--mode",
        choices=["nucleic-acid", "protein"],
        default="nucleic-acid",
        help="Primary molecule type to keep.",
    )
    parser.add_argument(
        "--keep-ligands",
        action="store_true",
        help="Keep non-water, non-ion ligands.",
    )
    parser.add_argument(
        "--keep-waters",
        action="store_true",
        help="Keep water residues.",
    )
    parser.add_argument(
        "--keep-ions",
        action="store_true",
        help="Keep ion residues.",
    )
    parser.add_argument(
        "--keep-altlocs",
        action="store_true",
        help="Keep non-primary alternate locations.",
    )
    parser.add_argument(
        "--chains",
        type=str,
        help="Comma-separated chain IDs to keep (e.g., A,B,C).",
    )
    parser.add_argument(
        "--model",
        type=int,
        help="Model number to keep.",
    )
    parser.add_argument("path", help="Path to a PDB or mmCIF file (optionally .gz).")
    args = parser.parse_args()

    chains = None
    if args.chains:
        chains = [chain.strip() for chain in args.chains.split(",") if chain.strip()]

    output = filter_file(
        args.path,
        mode=args.mode,
        keep_ligands=args.keep_ligands,
        keep_waters=args.keep_waters,
        keep_ions=args.keep_ions,
        keep_altlocs=args.keep_altlocs,
        chains=chains,
        model=args.model,
    )

    print(output, end="")


if __name__ == "__main__":
    main()
