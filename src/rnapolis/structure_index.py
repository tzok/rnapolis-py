"""Lightweight residue index for the adapter conversion pipeline."""

from dataclasses import dataclass, field
from typing import IO, Dict, List, Optional, Set, Tuple, Union

from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.parser import (
    _mmcif_value,
    _read_mmcif_data,
    detect_one_letter_name_from_names,
    get_one_letter_name,
    get_residue_name,
    is_cif,
    parse_pdb,
    try_parse_int,
)
from rnapolis.tertiary import Atom, Residue3D, is_nucleotide_from_atoms

CRITICAL_ATOM_NAMES = frozenset({"P", "O5'", "C1'", "N9", "N1"})

ResidueKey = Tuple[Optional[ResidueLabel], Optional[ResidueAuth], int]


@dataclass
class StructureIndex:
    """Residue-level structure representation used by external-tool adapters."""

    residues: List[Residue3D]
    residue_map: Dict[Union[ResidueLabel, ResidueAuth], Residue3D] = field(init=False)

    def __post_init__(self):
        """Populate the label/auth residue lookup table."""
        self.residue_map = {}
        for residue in self.residues:
            if residue.label is not None:
                self.residue_map[residue.label] = residue
            if residue.auth is not None:
                self.residue_map[residue.auth] = residue

    def find_residue(
        self,
        label: Optional[ResidueLabel],
        auth: Optional[ResidueAuth],
    ) -> Optional[Residue3D]:
        """Find a residue by label or author-style identifier."""
        if label is not None and label in self.residue_map:
            return self.residue_map.get(label)
        if auth is not None and auth in self.residue_map:
            return self.residue_map.get(auth)
        return None


def _store_critical_atom(
    atoms: Dict[str, Atom],
    atom_name: str,
    atom: Atom,
) -> None:
    """Keep the highest-occupancy atom for duplicate critical atom names."""
    existing = atoms.get(atom_name)
    if existing is None:
        atoms[atom_name] = atom
        return
    if atom.occupancy is not None and (
        existing.occupancy is None or atom.occupancy > existing.occupancy
    ):
        atoms[atom_name] = atom


def _build_residue(
    key: ResidueKey,
    entity_id: Optional[str],
    atom_names: Set[str],
    critical_atoms: Dict[str, Atom],
    modified: Dict[Union[ResidueAuth, ResidueLabel], str],
    sequence_by_entity: Dict[str, str],
) -> Residue3D:
    """Build a lightweight Residue3D from collected atom-level data."""
    label, auth, model = key
    residue_name = get_residue_name(auth, label, modified)
    one_letter_name = get_one_letter_name(
        entity_id, label, sequence_by_entity, residue_name
    )
    if one_letter_name not in "ACGUTN":
        one_letter_name = detect_one_letter_name_from_names(atom_names)
    nucleotide = is_nucleotide_from_atoms(
        atom_names, lambda atom_name: critical_atoms.get(atom_name)
    )
    return Residue3D(
        label,
        auth,
        model,
        one_letter_name,
        (),
        residue_name,
        nucleotide_hint=nucleotide,
    )


def read_structure_index(
    cif_or_pdb: IO[str],
    model: Optional[int] = None,
) -> StructureIndex:
    """Read only the residue-level data needed by external-tool adapters.

    Args:
        cif_or_pdb: Open file-like object with PDB or mmCIF content.
        model: Model number to extract (1-based). If None, the first model is used.

    Returns:
        A lightweight StructureIndex with identity-only residues.
    """
    if is_cif(cif_or_pdb):
        return _read_cif_index(cif_or_pdb, model)
    return _read_pdb_index(cif_or_pdb, model)


def _read_cif_index(cif_or_pdb: IO[str], model: Optional[int]) -> StructureIndex:
    data = _read_mmcif_data(cif_or_pdb)
    if not data:
        return StructureIndex([])

    block = data[0]
    atom_site = block.getObj("atom_site")
    if not atom_site:
        return StructureIndex([])

    modified: Dict[Union[ResidueLabel, ResidueAuth], str] = {}
    sequence_by_entity: Dict[str, str] = {}
    _read_cif_metadata(block, modified, sequence_by_entity)

    attributes = atom_site.getAttributeList()
    columns = {name: index for index, name in enumerate(attributes)}
    rows = atom_site.getRowList()

    available_models: List[int] = []
    for row in rows:
        current_model = _row_model(row, columns)
        if current_model not in available_models:
            available_models.append(current_model)
    if not available_models:
        return StructureIndex([])
    selected_model = model if model in available_models else available_models[0]

    residues: List[Residue3D] = []
    current_key: Optional[ResidueKey] = None
    current_entity_id: Optional[str] = None
    current_atom_names: Set[str] = set()
    current_critical_atoms: Dict[str, Atom] = {}

    def flush() -> None:
        nonlocal current_key
        if current_key is None:
            return
        residues.append(
            _build_residue(
                current_key,
                current_entity_id,
                current_atom_names,
                current_critical_atoms,
                modified,
                sequence_by_entity,
            )
        )
        current_key = None

    for row in rows:
        if _row_model(row, columns) != selected_model:
            continue

        label_chain = _mmcif_value(_row_value(row, columns, "label_asym_id"))
        label_number = try_parse_int(
            _mmcif_value(_row_value(row, columns, "label_seq_id"))
        )
        label_name = _mmcif_value(_row_value(row, columns, "label_comp_id"))
        auth_chain = _mmcif_value(_row_value(row, columns, "auth_asym_id"))
        auth_number = try_parse_int(
            _mmcif_value(_row_value(row, columns, "auth_seq_id"))
        )
        auth_name = _mmcif_value(_row_value(row, columns, "auth_comp_id"))
        insertion_code = _mmcif_value(_row_value(row, columns, "pdbx_PDB_ins_code"))

        if label_chain is None and auth_chain is None:
            continue
        if label_number is None and auth_number is None:
            continue
        if label_name is None and auth_name is None:
            continue

        label = (
            ResidueLabel(label_chain, label_number, label_name)
            if label_number is not None and label_name is not None
            else None
        )
        auth = (
            ResidueAuth(auth_chain, auth_number, insertion_code, auth_name)
            if auth_number is not None and auth_name is not None
            else None
        )
        if label is None and auth is None:
            continue

        key: ResidueKey = (label, auth, selected_model)
        if key != current_key:
            flush()
            current_key = key
            current_entity_id = None
            current_atom_names = set()
            current_critical_atoms = {}

        entity_id = _mmcif_value(_row_value(row, columns, "label_entity_id"))
        current_entity_id = entity_id
        atom_name = _row_value(row, columns, "label_atom_id")
        if atom_name is None:
            continue
        current_atom_names.add(atom_name)

        if atom_name in CRITICAL_ATOM_NAMES:
            atom = Atom(
                entity_id,
                label,
                auth,
                selected_model,
                atom_name,
                float(_row_value(row, columns, "Cartn_x")),
                float(_row_value(row, columns, "Cartn_y")),
                float(_row_value(row, columns, "Cartn_z")),
                _row_occupancy(row, columns),
            )
            _store_critical_atom(current_critical_atoms, atom_name, atom)

    flush()
    return StructureIndex(residues)


def _read_cif_metadata(
    block,
    modified: Dict[Union[ResidueLabel, ResidueAuth], str],
    sequence_by_entity: Dict[str, str],
) -> None:
    """Read modified-residue and entity-sequence metadata from an mmCIF block."""
    mod_residue = block.getObj("pdbx_struct_mod_residue")
    if mod_residue:
        attributes = mod_residue.getAttributeList()
        columns = {name: index for index, name in enumerate(attributes)}
        for row in mod_residue.getRowList():
            label_chain = _mmcif_value(_row_value(row, columns, "label_asym_id"))
            label_number = try_parse_int(
                _mmcif_value(_row_value(row, columns, "label_seq_id"))
            )
            label_name = _mmcif_value(_row_value(row, columns, "label_comp_id"))
            auth_chain = _mmcif_value(_row_value(row, columns, "auth_asym_id"))
            auth_number = try_parse_int(
                _mmcif_value(_row_value(row, columns, "auth_seq_id"))
            )
            auth_name = _mmcif_value(_row_value(row, columns, "auth_comp_id"))
            insertion_code = _mmcif_value(_row_value(row, columns, "PDB_ins_code"))
            label = (
                ResidueLabel(label_chain, label_number, label_name)
                if label_number is not None and label_name is not None
                else None
            )
            auth = (
                ResidueAuth(auth_chain, auth_number, insertion_code, auth_name)
                if auth_number is not None and auth_name is not None
                else None
            )
            standard_name = (
                _mmcif_value(_row_value(row, columns, "parent_comp_id")) or "n"
            )
            if label is not None:
                modified[label] = standard_name
            if auth is not None:
                modified[auth] = standard_name

    entity_poly = block.getObj("entity_poly")
    if entity_poly:
        attributes = entity_poly.getAttributeList()
        columns = {name: index for index, name in enumerate(attributes)}
        for row in entity_poly.getRowList():
            entity_id = _mmcif_value(_row_value(row, columns, "entity_id"))
            sequence = _mmcif_value(
                _row_value(row, columns, "pdbx_seq_one_letter_code_can")
            )
            if entity_id and sequence:
                sequence_by_entity[entity_id] = sequence.replace("\n", "")

    entity = block.getObj("entity")
    if entity:
        attributes = entity.getAttributeList()
        columns = {name: index for index, name in enumerate(attributes)}
        for row in entity.getRowList():
            entity_id = _mmcif_value(_row_value(row, columns, "id"))
            if entity_id:
                sequence_by_entity.setdefault(entity_id, "")


def _read_pdb_index(pdb: IO[str], model: Optional[int]) -> StructureIndex:
    atoms, modified, _, _ = parse_pdb(pdb)
    if not atoms:
        return StructureIndex([])

    available_models: List[int] = []
    for atom in atoms:
        if atom.model not in available_models:
            available_models.append(atom.model)
    selected_model = model if model in available_models else available_models[0]

    residues: List[Residue3D] = []
    current_key: Optional[ResidueKey] = None
    current_atom_names: Set[str] = set()
    current_critical_atoms: Dict[str, Atom] = {}

    def flush() -> None:
        nonlocal current_key
        if current_key is None:
            return
        residues.append(
            _build_residue(
                current_key,
                None,
                current_atom_names,
                current_critical_atoms,
                modified,
                {},
            )
        )
        current_key = None

    for atom in atoms:
        if atom.model != selected_model:
            continue
        key: ResidueKey = (atom.label, atom.auth, atom.model)
        if key != current_key:
            flush()
            current_key = key
            current_atom_names = set()
            current_critical_atoms = {}
        current_atom_names.add(atom.name)
        if atom.name in CRITICAL_ATOM_NAMES:
            _store_critical_atom(current_critical_atoms, atom.name, atom)

    flush()
    return StructureIndex(residues)


def _row_value(row, columns: Dict[str, int], name: str):
    """Return an mmCIF row value by column name, or None if absent."""
    index = columns.get(name)
    return row[index] if index is not None else None


def _row_model(row, columns: Dict[str, int]) -> int:
    """Return the model number for an mmCIF atom row."""
    value = _mmcif_value(_row_value(row, columns, "pdbx_PDB_model_num"))
    return int(value) if value is not None else 1


def _row_occupancy(row, columns: Dict[str, int]) -> Optional[float]:
    """Return occupancy for an mmCIF atom row, or None if missing."""
    value = _mmcif_value(_row_value(row, columns, "occupancy"))
    return float(value) if value is not None else None
