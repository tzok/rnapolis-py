import logging
import math
import string
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import IO, Dict, List, Optional, Set, Tuple, Union

import numpy
import numpy.typing
from mmcif.io.IoAdapterPy import IoAdapterPy

from rnapolis.common import (
    GlycosidicBond,
    LeontisWesthof,
    ResidueAuth,
    ResidueLabel,
)
from rnapolis.secondary import BasePair, Residue, Stacking

BASE_ATOMS = {
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"],
    "G": ["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "T": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"],
}

BASE_DONORS = {
    "A": ["C2", "N6", "C8", "O2'"],
    "G": ["N1", "N2", "C8", "O2'"],
    "C": ["N4", "C5", "C6", "O2'"],
    "U": ["N3", "C5", "C6", "O2'"],
    "T": ["N3", "C6", "C7"],
}

BASE_ACCEPTORS = {
    "A": ["N1", "N3", "N7"],
    "G": ["N3", "O6", "N7"],
    "C": ["O2", "N3"],
    "U": ["O2", "O4"],
    "T": ["O2", "O4"],
}

PHOSPHATE_ACCEPTORS = ["OP1", "OP2", "O5'", "O3'"]

RIBOSE_ACCEPTORS = ["O4'"]

BASE_EDGES = {
    "A": {
        "N1": "W",
        "C2": "WS",
        "N3": "S",
        "N6": "WH",
        "N7": "H",
        "C8": "H",
        "O2'": "S",
    },
    "G": {
        "N1": "W",
        "N2": "WS",
        "N3": "S",
        "O6": "WH",
        "N7": "H",
        "C8": "H",
        "O2'": "S",
    },
    "C": {
        "O2": "WS",
        "N3": "W",
        "N4": "WH",
        "C5": "H",
        "C6": "H",
        "O2'": "S",
    },
    "U": {
        "O2": "WS",
        "N3": "W",
        "O4": "WH",
        "C5": "H",
        "C6": "H",
        "O2'": "S",
    },
    "T": {
        "O2": "WS",
        "N3": "W",
        "O4": "WH",
        "C6": "H",
        "C7": "H",
    },
}


@dataclass(frozen=True)
class Atom:
    label: Optional[ResidueLabel]
    auth: Optional[ResidueAuth]
    model: int
    name: str
    x: float
    y: float
    z: float

    @property
    def coordinates(self) -> numpy.typing.NDArray[numpy.floating]:
        return numpy.array([self.x, self.y, self.z])


@dataclass(order=True)
class Residue3D(Residue):
    index: int
    model: int
    one_letter_name: str
    atoms: Tuple[Atom, ...]

    # Dict representing expected name of atom involved in glycosidic bond
    outermost_atoms = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "T": "N1"}
    # Dist representing expected name of atom closest to the tetrad center
    innermost_atoms = {"A": "N6", "G": "O6", "C": "N4", "U": "O4", "T": "O4"}

    def __hash__(self):
        return hash((self.name, self.model, self.label, self.auth, self.atoms))

    def __repr__(self):
        return f"{self.full_name}"

    @property
    def chi(self) -> float:
        if self.one_letter_name.upper() in ("A", "G"):
            return self.__chi_purine()
        elif self.one_letter_name.upper() in ("C", "U", "T"):
            return self.__chi_pyrimidine()
        # if unknown, try purine first, then pyrimidine
        torsion = self.__chi_purine()
        if math.isnan(torsion):
            return self.__chi_pyrimidine()
        return torsion

    # TODO: the ranges could be modified to match Saenger
    @property
    def chi_class(self) -> Optional[GlycosidicBond]:
        if math.isnan(self.chi):
            return None
        if -math.pi / 2 < self.chi < math.pi / 2:
            return GlycosidicBond.syn
        return GlycosidicBond.anti

    @property
    def outermost_atom(self) -> Atom:
        return next(filter(None, self.__outer_generator()))

    @property
    def innermost_atom(self) -> Atom:
        return next(filter(None, self.__inner_generator()))

    @property
    def is_nucleotide(self) -> bool:
        return len(self.atoms) > 1 and any(
            [atom for atom in self.atoms if atom.name == "C1'"]
        )

    @property
    def base_normal_vector(self) -> Optional[numpy.typing.NDArray[numpy.floating]]:
        if self.one_letter_name in "AG":
            n9 = self.find_atom("N9")
            n7 = self.find_atom("N7")
            n3 = self.find_atom("N3")
            if n9 is None or n7 is None or n3 is None:
                return None
            v1 = n7.coordinates - n9.coordinates
            v2 = n3.coordinates - n9.coordinates
        else:
            n1 = self.find_atom("N1")
            c4 = self.find_atom("C4")
            o2 = self.find_atom("O2")
            if n1 is None or c4 is None or o2 is None:
                return None
            v1 = c4.coordinates - n1.coordinates
            v2 = o2.coordinates - n1.coordinates
        normal: numpy.typing.NDArray[numpy.floating] = numpy.cross(v1, v2)
        return normal / numpy.linalg.norm(normal)

    def find_atom(self, atom_name: str) -> Optional[Atom]:
        for atom in self.atoms:
            if atom.name == atom_name:
                return atom
        return None

    def __chi_purine(self) -> float:
        atoms = [
            self.find_atom("O4'"),
            self.find_atom("C1'"),
            self.find_atom("N9"),
            self.find_atom("C4"),
        ]
        if all([atom is not None for atom in atoms]):
            return Residue3D.__torsion_angle(atoms)  # type: ignore
        return math.nan

    def __chi_pyrimidine(self) -> float:
        atoms = [
            self.find_atom("O4'"),
            self.find_atom("C1'"),
            self.find_atom("N1"),
            self.find_atom("C2"),
        ]
        if all([atom is not None for atom in atoms]):
            return Residue3D.__torsion_angle(atoms)  # type: ignore
        return math.nan

    def __outer_generator(self):
        # try to find expected atom name
        upper = self.one_letter_name.upper()
        if upper in self.outermost_atoms:
            yield self.find_atom(self.outermost_atoms[upper])

        # try to get generic name for purine/pyrimidine
        yield self.find_atom("N9")
        yield self.find_atom("N1")

        # try to find at least C1' next to nucleobase
        yield self.find_atom("C1'")

        # get any atom
        if self.atoms:
            yield self.atoms[0]

        # last resort, create pseudoatom at (0, 0, 0)
        logging.error(
            f"Failed to determine the outermost atom for nucleotide {self}, so an arbitrary atom will be used"
        )
        yield Atom(self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0)

    def __inner_generator(self):
        # try to find expected atom name
        upper = self.one_letter_name.upper()
        if upper in self.innermost_atoms:
            yield self.find_atom(self.innermost_atoms[upper])

        # try to get generic name for purine/pyrimidine
        yield self.find_atom("C6")
        yield self.find_atom("C4")

        # try to find any atom at position 4 or 6 for purine/pyrimidine respectively
        yield self.find_atom("O6")
        yield self.find_atom("N6")
        yield self.find_atom("S6")
        yield self.find_atom("O4")
        yield self.find_atom("N4")
        yield self.find_atom("S4")

        # get any atom
        if self.atoms:
            yield self.atoms[0]

        # last resort, create pseudoatom at (0, 0, 0)
        logging.error(
            f"Failed to determine the innermost atom for nucleotide {self}, so an arbitrary atom will be used"
        )
        yield Atom(self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0)


@dataclass
class BasePair3D(BasePair):
    nt1_3d: Residue3D
    nt2_3d: Residue3D

    score_table = {
        LeontisWesthof.cWW: 1,
        LeontisWesthof.tWW: 2,
        LeontisWesthof.cWH: 3,
        LeontisWesthof.tWH: 4,
        LeontisWesthof.cWS: 5,
        LeontisWesthof.tWS: 6,
        LeontisWesthof.cHW: 7,
        LeontisWesthof.tHW: 8,
        LeontisWesthof.cHH: 9,
        LeontisWesthof.tHH: 10,
        LeontisWesthof.cHS: 11,
        LeontisWesthof.tHS: 12,
        LeontisWesthof.cSW: 13,
        LeontisWesthof.tSW: 14,
        LeontisWesthof.cSH: 15,
        LeontisWesthof.tSH: 16,
        LeontisWesthof.cSS: 17,
        LeontisWesthof.tSS: 18,
    }

    @property
    def reverse(self):
        lw = f"{self.lw.name[0]}{self.lw.name[2]}{self.lw.name[1]}"
        return BasePair3D(
            self.nt2,
            self.nt1,
            LeontisWesthof[lw],
            self.saenger,
            self.nt2_3d,
            self.nt1_3d,
        )

    @property
    def score(self) -> int:
        return self.score_table.get(self.lw, 20)

    @property
    def is_canonical(self) -> bool:
        if self.saenger is not None:
            return self.saenger.is_canonical

        nts = "".join(
            sorted(
                [
                    self.nt1_3d.one_letter_name.upper(),
                    self.nt2_3d.one_letter_name.upper(),
                ]
            )
        )
        return self.lw == LeontisWesthof.cWW and (
            nts == "AU" or nts == "AT" or nts == "CG" or nts == "GU"
        )

    def conflicts_with(self, other) -> bool:
        xi, yi = sorted([self.nt1_3d.index, self.nt2_3d.index])
        xj, yj = sorted([other.nt1_3d.index, other.nt2_3d.index])
        return xi < xj < yi < yj or xj < xi < yj < yi

    def in_tetrad(self, analysis) -> bool:
        for tetrad in analysis.tetrads:
            if self in (
                tetrad.pair_12,
                tetrad.pair_23,
                tetrad.pair_34,
                tetrad.pair_41,
                tetrad.pair_12.reverse(),
                tetrad.pair_23.reverse(),
                tetrad.pair_34.reverse(),
                tetrad.pair_41.reverse(),
            ):
                return True
        return False


@dataclass
class Stacking3D(Stacking):
    nt1_3d: Residue3D
    nt2_3d: Residue3D


@dataclass
class Structure3D:
    residues: List[Residue3D]
    residue_map: Dict[Union[ResidueLabel, ResidueAuth], Residue3D] = field(init=False)

    def __post_init__(self):
        self.residue_map = {}
        for residue in self.residues:
            if residue.label is not None:
                self.residue_map[residue.label] = residue
            if residue.auth is not None:
                self.residue_map[residue.auth] = residue

    def find_residue(
        self, label: Optional[ResidueLabel], auth: Optional[ResidueAuth]
    ) -> Optional[Residue3D]:
        if label is not None and label in self.residue_map:
            return self.residue_map.get(label)
        if auth is not None and auth in self.residue_map:
            return self.residue_map.get(auth)
        return None

    # FIXME
    # def base_pairs(self, structure2d: Structure2D) -> List[BasePair3D]:
    #     result = []
    #     for base_pair in structure2d.basePairs:
    #         nt1 = self.find_residue(base_pair.nt1.label, base_pair.nt1.auth)
    #         nt2 = self.find_residue(base_pair.nt2.label, base_pair.nt2.auth)
    #         if nt1 is not None and nt2 is not None:
    #             result.append(BasePair3D(nt1, nt2, base_pair.lw))
    #     return result

    # FIXME
    # def base_pair_graph(
    #     self, structure2d: Structure2D, strict: bool = False
    # ) -> Dict[Residue3D, Set[Residue3D]]:
    #     graph = defaultdict(set)
    #     for pair in self.base_pairs(structure2d):
    #         if strict and pair.lw not in (LeontisWesthof.cWH, LeontisWesthof.cHW):
    #             continue
    #         graph[pair.nt1].add(pair.nt2)
    #         graph[pair.nt2].add(pair.nt1)
    #     return graph

    # FIXME
    # def base_pair_dict(
    #     self, structure2d: Structure2D, strict: bool = False
    # ) -> Dict[Tuple[Residue3D, Residue3D], BasePair3D]:
    #     result = {}
    #     for pair in self.base_pairs(structure2d):
    #         if strict and pair.lw not in (LeontisWesthof.cWH, LeontisWesthof.cHW):
    #             continue
    #         result[(pair.nt1, pair.nt2)] = pair
    #     return result

    # FIXME
    # def stackings(self, structure2d: Structure2D) -> List[Stacking3D]:
    #     result = []
    #     for stacking in structure2d.stackings:
    #         nt1 = self.find_residue(stacking.nt1.label, stacking.nt1.auth)
    #         nt2 = self.find_residue(stacking.nt2.label, stacking.nt2.auth)
    #         if nt1 is not None and nt2 is not None:
    #             result.append(Stacking3D(nt1, nt2))
    #     return result

    # FIXME
    # def stacking_graph(
    #     self, structure2d: Structure2D
    # ) -> Dict[Residue3D, Set[Residue3D]]:
    #     graph = defaultdict(set)
    #     for pair in self.stackings(structure2d):
    #         graph[pair.nt1].add(pair.nt2)
    #         graph[pair.nt2].add(pair.nt1)
    #     return graph

    # def stacking_dict(
    #     self, structure2d: Structure2D
    # ) -> Dict[Tuple[Residue3D, Residue3D], Stacking3D]:
    #     result = {}
    #     for pair in self.stackings(structure2d):
    #         result[(pair.nt1, pair.nt2)] = pair
    #     return result


def read_3d_structure(cif_or_pdb: IO[str], model: int) -> Structure3D:
    atoms, modified, sequence = (
        parse_cif(cif_or_pdb) if is_cif(cif_or_pdb) else parse_pdb(cif_or_pdb)
    )
    atoms = list(filter(lambda atom: atom.model == model, atoms))
    return group_atoms(atoms, modified, sequence)


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
                if label_chain_name and label_residue_number and label_residue_name:
                    label = ResidueLabel(
                        label_chain_name, label_residue_number, label_residue_name
                    )

                auth = None
                if auth_chain_name and auth_residue_number and auth_residue_name:
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
                atoms.append(Atom(label, auth, model, atom_name, x, y, z))

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
                if label_chain_name and label_residue_number and label_residue_name:
                    label = ResidueLabel(
                        label_chain_name, label_residue_number, label_residue_name
                    )

                auth = None
                if (
                    auth_chain_name
                    and auth_residue_number
                    and auth_residue_name
                    and insertion_code
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
            auth = ResidueAuth(
                chain_identifier, residue_number, insertion_code, residue_name
            )
            atoms.append(Atom(None, auth, model, atom_name, x, y, z))
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
) -> Structure3D:
    if not atoms:
        return Structure3D([])

    key_previous = (atoms[0].label, atoms[0].auth, atoms[0].model)
    residue_atoms = [atoms[0]]
    residues: List[Residue3D] = []
    index = 1

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
            residues.append(
                Residue3D(
                    label, auth, index, model, one_letter_name, tuple(residue_atoms)
                )
            )
            index += 1
            key_previous = key
            residue_atoms = [atom]

    label = key_previous[0]
    auth = key_previous[1]
    model = key_previous[2]
    name = get_residue_name(auth, label, modified)
    one_letter_name = get_one_letter_name(label, sequence, name)
    if one_letter_name not in "ACGUT":
        one_letter_name = detect_one_letter_name(residue_atoms)
    residues.append(
        Residue3D(label, auth, index, model, one_letter_name, tuple(residue_atoms))
    )

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
