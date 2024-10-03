import logging
from typing import IO, Dict, List, Optional, Tuple, Union

import numpy as np
from mmcif.io.IoAdapterPy import IoAdapterPy
from scipy.spatial import KDTree

from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.tertiary import BASE_ATOMS, Atom, Residue3D, Structure3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_3d_structure(
    cif_or_pdb: IO[str], model: Optional[int] = None, nucleic_acid_only: bool = False
) -> Structure3D:
    atoms, modified, sequence_by_entity, is_nucleic_acid_by_entity = (
        parse_cif(cif_or_pdb) if is_cif(cif_or_pdb) else parse_pdb(cif_or_pdb)
    )
    available_models = {atom.model: None for atom in atoms}
    atoms_by_model = {
        model: list(filter(lambda atom: atom.model == model, atoms))
        for model in available_models
    }
    if model is not None and model in available_models:
        atoms = atoms_by_model[model]
    else:
        atoms = atoms_by_model[list(available_models.keys())[0]]
    return group_atoms(
        atoms,
        modified,
        sequence_by_entity,
        is_nucleic_acid_by_entity,
        nucleic_acid_only,
    )


def is_cif(cif_or_pdb: IO[str]) -> bool:
    cif_or_pdb.seek(0)
    for line in cif_or_pdb.readlines():
        if line.startswith("_atom_site"):
            return True
    return False


def parse_cif(
    cif: IO[str],
) -> Tuple[
    List[Atom],
    Dict[Union[ResidueLabel, ResidueAuth], str],
    Dict[str, str],
    Dict[str, bool],
]:
    cif.seek(0)

    io_adapter = IoAdapterPy()
    data = io_adapter.readFile(cif.name)
    atoms_to_process: List[Atom] = []
    modified: Dict[Union[ResidueLabel, ResidueAuth], str] = {}
    sequence_by_entity: Dict[str, str] = {}
    is_nucleic_acid_by_entity: Dict[str, bool] = {}

    if data:
        atom_site = data[0].getObj("atom_site")
        mod_residue = data[0].getObj("pdbx_struct_mod_residue")
        entity_poly = data[0].getObj("entity_poly")

        if atom_site:
            for row in atom_site.getRowList():
                row_dict = dict(zip(atom_site.getAttributeList(), row))

                label_entity_id = row_dict.get("label_entity_id", None)
                label_chain_name = row_dict.get("label_asym_id", None)
                label_residue_number = try_parse_int(row_dict.get("label_seq_id", None))
                label_residue_name = row_dict.get("label_comp_id", None)
                auth_chain_name = row_dict.get("auth_asym_id", None)
                auth_residue_number = try_parse_int(row_dict.get("auth_seq_id", None))
                auth_residue_name = row_dict.get("auth_comp_id", None)
                insertion_code = row_dict.get("pdbx_PDB_ins_code", None)

                # mmCIF marks empty values with ?
                if insertion_code == "?":
                    insertion_code = None

                if label_chain_name is None and auth_chain_name is None:
                    raise RuntimeError(
                        f"Cannot parse an atom line with empty chain name: {row}"
                    )
                if label_residue_number is None and auth_residue_number is None:
                    raise RuntimeError(
                        f"Cannot parse an atom line with empty residue number: {row}"
                    )
                if label_residue_name is None and auth_residue_name is None:
                    raise RuntimeError(
                        f"Cannot parse an atom line with empty residue name: {row}"
                    )

                label = None
                if (
                    label_chain_name is not None
                    and label_residue_number is not None
                    and label_residue_name is not None
                ):
                    label = ResidueLabel(
                        label_chain_name, label_residue_number, label_residue_name
                    )

                auth = None
                if (
                    auth_chain_name is not None
                    and auth_residue_number is not None
                    and auth_residue_name is not None
                ):
                    auth = ResidueAuth(
                        auth_chain_name,
                        auth_residue_number,
                        insertion_code,
                        auth_residue_name,
                    )

                if label is None and auth is None:
                    # this should not happen in a valid mmCIF file
                    # skipping the line
                    logger.debug(
                        f"Cannot parse an atom line without chain name, residue number, and residue name: {row}"
                    )
                    continue

                model = int(row_dict.get("pdbx_PDB_model_num", "1"))
                atom_name = row_dict["label_atom_id"]
                x = float(row_dict["Cartn_x"])
                y = float(row_dict["Cartn_y"])
                z = float(row_dict["Cartn_z"])

                occupancy = (
                    float(row_dict["occupancy"])
                    if "occupancy" in row_dict and row_dict["occupancy"] != "."
                    else None
                )

                atoms_to_process.append(
                    Atom(
                        label_entity_id,
                        label,
                        auth,
                        model,
                        atom_name,
                        x,
                        y,
                        z,
                        occupancy,
                    )
                )

        if mod_residue:
            for row in mod_residue.getRowList():
                row_dict = dict(zip(mod_residue.getAttributeList(), row))

                label_chain_name = row_dict.get("label_asym_id", None)
                label_residue_number = try_parse_int(row_dict.get("label_seq_id", None))
                label_residue_name = row_dict.get("label_comp_id", None)
                auth_chain_name = row_dict.get("auth_asym_id", None)
                auth_residue_number = try_parse_int(row_dict.get("auth_seq_id", None))
                auth_residue_name = row_dict.get("auth_comp_id", None)
                insertion_code = row_dict.get("PDB_ins_code", None)

                label = None
                if (
                    label_chain_name is not None
                    and label_residue_number is not None
                    and label_residue_name is not None
                ):
                    label = ResidueLabel(
                        label_chain_name, label_residue_number, label_residue_name
                    )

                auth = None
                if (
                    auth_chain_name is not None
                    and auth_residue_number is not None
                    and auth_residue_name is not None
                    and insertion_code is not None
                ):
                    auth = ResidueAuth(
                        auth_chain_name,
                        auth_residue_number,
                        insertion_code,
                        auth_residue_name,
                    )

                # TODO: is processing this data for each model separately required?
                # model = row_dict.get('PDB_model_num', '1')
                standard_residue_name = row_dict.get("parent_comp_id", "n")

                if label is not None:
                    modified[label] = standard_residue_name
                if auth is not None:
                    modified[auth] = standard_residue_name

        if entity_poly:
            for row in entity_poly.getRowList():
                row_dict = dict(zip(entity_poly.getAttributeList(), row))

                entity_id = row_dict.get("entity_id", None)
                type_ = row_dict.get("type", None)
                pdbx_seq_one_letter_code_can = row_dict.get(
                    "pdbx_seq_one_letter_code_can", None
                )

                if entity_id and type_:
                    is_nucleic_acid_by_entity[entity_id] = type_ in (
                        "peptide nucleic acid",
                        "polydeoxyribonucleotide",
                        "polydeoxyribonucleotide/polyribonucleotide hybrid",
                        "polyribonucleotide",
                    )

                if entity_id and pdbx_seq_one_letter_code_can:
                    sequence_by_entity[entity_id] = pdbx_seq_one_letter_code_can

    atoms = filter_clashing_atoms(atoms_to_process)
    return atoms, modified, sequence_by_entity, is_nucleic_acid_by_entity


def parse_pdb(
    pdb: IO[str],
) -> Tuple[
    List[Atom],
    Dict[Union[ResidueLabel, ResidueAuth], str],
    Dict[str, str],
    Dict[str, bool],
]:
    pdb.seek(0)
    atoms_to_process: List[Atom] = []
    modified: Dict[Union[ResidueLabel, ResidueAuth], str] = {}
    model = 1

    for line in pdb.readlines():
        if line.startswith("MODEL"):
            model = int(line[10:14].strip())
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_identifier = line[21]
            residue_number = int(line[22:26].strip())
            insertion_code = line[26] if line[26] != " " else None
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occupancy = float(line[54:60].strip())
            auth = ResidueAuth(
                chain_identifier, residue_number, insertion_code, residue_name
            )

            atoms_to_process.append(
                Atom(None, None, auth, model, atom_name, x, y, z, occupancy)
            )
        elif line.startswith("MODRES"):
            original_name = line[12:15]
            chain_identifier = line[16]
            residue_number = int(line[18:22].strip())
            insertion_code = line[23]
            standard_residue_name = line[24:27].strip()
            auth = ResidueAuth(
                chain_identifier, residue_number, insertion_code, original_name
            )
            modified[auth] = standard_residue_name

    atoms = filter_clashing_atoms(atoms_to_process)
    return atoms, modified, {}, {}


def group_atoms(
    atoms: List[Atom],
    modified: Dict[Union[ResidueLabel, ResidueAuth], str],
    sequence_by_entity: Dict[str, str],
    is_nucleic_acid_by_entity: Dict[str, bool],
    nucleic_acid_only: bool,
) -> Structure3D:
    if not atoms:
        return Structure3D([])

    key_previous = (atoms[0].label, atoms[0].auth, atoms[0].model)
    residue_atoms = [atoms[0]]
    residues: List[Residue3D] = []

    for atom in atoms[1:]:
        key = (atom.label, atom.auth, atom.model)
        if key == key_previous:
            residue_atoms.append(atom)
        else:
            label = key_previous[0]
            auth = key_previous[1]
            model = key_previous[2]
            entity_id = residue_atoms[-1].entity_id
            name = get_residue_name(auth, label, modified)
            one_letter_name = get_one_letter_name(
                entity_id, label, sequence_by_entity, name
            )

            if one_letter_name not in "ACGUTN":
                one_letter_name = detect_one_letter_name(residue_atoms)

            residues.append(
                Residue3D(label, auth, model, one_letter_name, tuple(residue_atoms))
            )

            key_previous = key
            residue_atoms = [atom]

    label = key_previous[0]
    auth = key_previous[1]
    model = key_previous[2]
    entity_id = residue_atoms[-1].entity_id
    name = get_residue_name(auth, label, modified)
    one_letter_name = get_one_letter_name(entity_id, label, sequence_by_entity, name)

    if one_letter_name not in "ACGUTN":
        one_letter_name = detect_one_letter_name(residue_atoms)

    residues.append(
        Residue3D(label, auth, model, one_letter_name, tuple(residue_atoms))
    )

    if nucleic_acid_only:
        if is_nucleic_acid_by_entity:
            residues = [
                residue
                for residue in residues
                if is_nucleic_acid_by_entity[residue.atoms[0].entity_id]
            ]
        else:
            residues = [residue for residue in residues if residue.is_nucleotide]

    return Structure3D(residues)


def get_residue_name(
    auth: Optional[ResidueAuth],
    label: Optional[ResidueLabel],
    modified: Dict[Union[ResidueAuth, ResidueLabel], str],
) -> str:
    if auth is not None and auth in modified:
        name = modified[auth].lower()
    elif label is not None and label in modified:
        name = modified[label].lower()
    elif auth is not None:
        name = auth.name
    elif label is not None:
        name = label.name
    else:
        # any nucleotide
        name = "n"
    return name


def get_one_letter_name(
    entity_id: Optional[str],
    label: Optional[ResidueLabel],
    sequence_by_entity: Dict[str, str],
    name: str,
) -> str:
    # try getting the value from _entity_poly first
    if entity_id is not None and label is not None and entity_id in sequence_by_entity:
        return sequence_by_entity[entity_id][label.number - 1]
    # RNA
    if len(name) == 1:
        return name
    # DNA
    if len(name) == 2 and name[0].upper() == "D":
        return name[1]
    # try the last letter of the name
    if str.isalpha(name[-1]):
        return name[-1]
    # any nucleotide
    return "n"


def detect_one_letter_name(atoms: List[Atom]) -> str:
    atom_names_present = {atom.name for atom in atoms}
    score = {}
    for candidate in "ACGUT":
        atom_names_expected = BASE_ATOMS[candidate]
        count = sum(
            1 for atom in atom_names_expected if atom in atom_names_present
        ) / len(atom_names_expected)
        score[candidate] = count
    items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    if items[0][1] == 0:
        return "?"
    return items[0][0]


def try_parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None


def filter_clashing_atoms(atoms: List[Atom], clash_distance: float = 0.5) -> List[Atom]:
    # First, remove duplicate atoms
    unique_atoms = {}

    for i, atom in enumerate(atoms):
        key = (atom.label, atom.auth, atom.name)
        if key not in unique_atoms or atom.occupancy > unique_atoms[key].occupancy:
            unique_atoms[key] = atom

    unique_atoms_list = list(unique_atoms.values())

    # Now handle clashing atoms
    coords = np.array([(atom.x, atom.y, atom.z) for atom in unique_atoms_list])
    tree = KDTree(coords)

    pairs = tree.query_pairs(r=clash_distance)

    atoms_to_keep = set(range(len(unique_atoms_list)))

    for i, j in pairs:
        if (
            unique_atoms_list[i].occupancy is None
            or unique_atoms_list[j].occupancy is None
        ):
            continue
        if unique_atoms_list[i].occupancy > unique_atoms_list[j].occupancy:
            atoms_to_keep.discard(j)
        else:
            atoms_to_keep.discard(i)

    return [unique_atoms_list[i] for i in atoms_to_keep]
