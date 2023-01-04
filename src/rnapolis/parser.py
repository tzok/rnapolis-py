from typing import IO, Dict, List, Optional, Tuple, Union

from mmcif.io.IoAdapterPy import IoAdapterPy

from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.tertiary import BASE_ATOMS, Atom, Residue3D, Structure3D


def read_3d_structure(
    cif_or_pdb: IO[str], model: int = 1, nucleic_acid_only: bool = False
) -> Structure3D:
    atoms, modified, sequence = (
        parse_cif(cif_or_pdb) if is_cif(cif_or_pdb) else parse_pdb(cif_or_pdb)
    )
    atoms = list(filter(lambda atom: atom.model == model, atoms))
    return group_atoms(atoms, modified, sequence, nucleic_acid_only)


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
    Dict[Tuple[str, int], str],
]:
    cif.seek(0)

    io_adapter = IoAdapterPy()
    data = io_adapter.readFile(cif.name)
    atoms: List[Atom] = []
    modified: Dict[Union[ResidueLabel, ResidueAuth], str] = {}
    sequence = {}

    if data:
        atom_site = data[0].getObj("atom_site")
        mod_residue = data[0].getObj("pdbx_struct_mod_residue")
        entity_poly = data[0].getObj("entity_poly")

        if atom_site:
            for row in atom_site.getRowList():
                row_dict = dict(zip(atom_site.getAttributeList(), row))

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

                atoms.append(Atom(label, auth, model, atom_name, x, y, z, occupancy))

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

                pdbx_strand_id = row_dict.get("pdbx_strand_id", None)
                pdbx_seq_one_letter_code_can = row_dict.get(
                    "pdbx_seq_one_letter_code_can", None
                )

                if pdbx_strand_id and pdbx_seq_one_letter_code_can:
                    for strand in pdbx_strand_id.split(","):
                        for i, letter in enumerate(pdbx_seq_one_letter_code_can):
                            sequence[(strand, i + 1)] = letter

    return atoms, modified, sequence


def parse_pdb(
    pdb: IO[str],
) -> Tuple[
    List[Atom],
    Dict[Union[ResidueLabel, ResidueAuth], str],
    Dict[Tuple[str, int], str],
]:
    pdb.seek(0)
    atoms: List[Atom] = []
    modified: Dict[Union[ResidueLabel, ResidueAuth], str] = {}
    model = 1

    for line in pdb.readlines():
        if line.startswith("MODEL"):
            model = int(line[10:14].strip())
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            alternate_location = line[16]
            if alternate_location != " ":
                continue
            atom_name = line[12:16].strip()
            residue_name = line[18:20].strip()
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
            atoms.append(Atom(None, auth, model, atom_name, x, y, z, occupancy))
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

    return atoms, modified, {}


def group_atoms(
    atoms: List[Atom],
    modified: Dict[Union[ResidueLabel, ResidueAuth], str],
    sequence: Dict[Tuple[str, int], str],
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
            name = get_residue_name(auth, label, modified)
            one_letter_name = get_one_letter_name(label, sequence, name)
            if one_letter_name not in "ACGUT":
                one_letter_name = detect_one_letter_name(residue_atoms)
            residue = Residue3D(
                label, auth, model, one_letter_name, tuple(residue_atoms)
            )
            if not nucleic_acid_only or (nucleic_acid_only and residue.is_nucleotide):
                residues.append(residue)
            key_previous = key
            residue_atoms = [atom]

    label = key_previous[0]
    auth = key_previous[1]
    model = key_previous[2]
    name = get_residue_name(auth, label, modified)
    one_letter_name = get_one_letter_name(label, sequence, name)
    if one_letter_name not in "ACGUT":
        one_letter_name = detect_one_letter_name(residue_atoms)
    residue = Residue3D(label, auth, model, one_letter_name, tuple(residue_atoms))
    if not nucleic_acid_only or (nucleic_acid_only and residue.is_nucleotide):
        residues.append(residue)

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
    label: Optional[ResidueLabel], sequence: Dict[Tuple[str, int], str], name: str
) -> str:
    # try getting the value from _entity_poly first
    if label is not None:
        key = (label.chain, label.number)
        if key in sequence:
            return sequence[key]

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
    return items[0][0]


def try_parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except:
        return None
