import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, total_ordering
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy
import numpy.typing

from rnapolis.common import (
    BasePair,
    BpSeq,
    Entry,
    GlycosidicBond,
    LeontisWesthof,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Stacking,
)

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

RIBOSE_ACCEPTORS = ["O4'", "O2'"]

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

AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT = 1.6


@dataclass(frozen=True, order=True)
class Atom:
    label: Optional[ResidueLabel]
    auth: Optional[ResidueAuth]
    model: int
    name: str
    x: float
    y: float
    z: float
    occupancy: Optional[float]

    @cached_property
    def coordinates(self) -> numpy.typing.NDArray[numpy.floating]:
        return numpy.array([self.x, self.y, self.z])


@dataclass(frozen=True)
@total_ordering
class Residue3D(Residue):
    model: int
    one_letter_name: str
    atoms: Tuple[Atom, ...]

    # Dict representing expected name of atom involved in glycosidic bond
    outermost_atoms = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "T": "N1"}
    # Dist representing expected name of atom closest to the tetrad center
    innermost_atoms = {"A": "N6", "G": "O6", "C": "N4", "U": "O4", "T": "O4"}
    # Heavy atoms for each main nucleobase
    nucleobase_heavy_atoms = {
        "A": set(["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"]),
        "G": set(["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"]),
        "C": set(["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]),
        "U": set(["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]),
    }

    def __lt__(self, other):
        return (self.model, self.chain, self.number, self.icode or " ") < (
            other.model,
            other.chain,
            other.number,
            other.icode or " ",
        )

    def __hash__(self):
        return hash((self.model, self.label, self.auth))

    def __repr__(self):
        return f"{self.full_name}"

    @cached_property
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

    @cached_property
    def chi_class(self) -> Optional[GlycosidicBond]:
        if math.isnan(self.chi):
            return None
        # syn is between -30 and 120 degress
        # this complies with Neidle "Principles of Nucleic Acid Structure" and with own research
        if math.radians(-30) < self.chi < math.radians(120):
            return GlycosidicBond.syn
        # the rest is anti
        return GlycosidicBond.anti

    @cached_property
    def outermost_atom(self) -> Atom:
        return next(filter(None, self.__outer_generator()))

    @cached_property
    def innermost_atom(self) -> Atom:
        return next(filter(None, self.__inner_generator()))

    @cached_property
    def is_nucleotide(self) -> bool:
        return len(self.atoms) > 1 and any(
            [atom for atom in self.atoms if atom.name == "C1'"]
        )

    @cached_property
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

    @cached_property
    def has_all_nucleobase_heavy_atoms(self) -> bool:
        if self.one_letter_name in "ACGU":
            present_atom_names = set([atom.name for atom in self.atoms])
            expected_atom_names = Residue3D.nucleobase_heavy_atoms[self.one_letter_name]
            return expected_atom_names.issubset(present_atom_names)
        return False

    def find_atom(self, atom_name: str) -> Optional[Atom]:
        for atom in self.atoms:
            if atom.name == atom_name:
                return atom
        return None

    def is_connected(self, next_residue_candidate) -> bool:
        o3p = self.find_atom("O3'")
        p = next_residue_candidate.find_atom("P")

        if o3p is not None and p is not None:
            distance = numpy.linalg.norm(o3p.coordinates - p.coordinates).item()
            return distance < 1.5 * AVERAGE_OXYGEN_PHOSPHORUS_DISTANCE_COVALENT

        return False

    def __chi_purine(self) -> float:
        atoms = [
            self.find_atom("O4'"),
            self.find_atom("C1'"),
            self.find_atom("N9"),
            self.find_atom("C4"),
        ]
        if all([atom is not None for atom in atoms]):
            return torsion_angle(*atoms)  # type: ignore
        return math.nan

    def __chi_pyrimidine(self) -> float:
        atoms = [
            self.find_atom("O4'"),
            self.find_atom("C1'"),
            self.find_atom("N1"),
            self.find_atom("C2"),
        ]
        if all([atom is not None for atom in atoms]):
            return torsion_angle(*atoms)  # type: ignore
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
        yield Atom(self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0, None)

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
        yield Atom(self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0, None)


@dataclass(frozen=True, order=True)
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

    @cached_property
    def reverse(self):
        return BasePair3D(
            self.nt2,
            self.nt1,
            self.lw.reverse,
            self.saenger,
            self.nt2_3d,
            self.nt1_3d,
        )

    @cached_property
    def score(self) -> int:
        return self.score_table.get(self.lw, 20)

    @cached_property
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


@dataclass(frozen=True, order=True)
class Stacking3D(Stacking):
    nt1_3d: Residue3D
    nt2_3d: Residue3D

    @cached_property
    def reverse(self):
        if self.topology is None:
            return self
        return Stacking3D(
            self.nt2, self.nt1, self.topology.reverse, self.nt2_3d, self.nt1_3d
        )


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


@dataclass
class Mapping2D3D:
    structure3d: Structure3D
    base_pairs2d: List[BasePair]
    stackings2d: List[Stacking]
    find_gaps: bool

    @cached_property
    def base_pairs(self) -> List[BasePair3D]:
        result = []
        used = set()
        for base_pair in self.base_pairs2d:
            nt1 = self.structure3d.find_residue(base_pair.nt1.label, base_pair.nt1.auth)
            nt2 = self.structure3d.find_residue(base_pair.nt2.label, base_pair.nt2.auth)
            if nt1 is not None and nt2 is not None:
                bp = BasePair3D(
                    base_pair.nt1,
                    base_pair.nt2,
                    base_pair.lw,
                    base_pair.saenger,
                    nt1,
                    nt2,
                )
                if bp not in used:
                    result.append(bp)
                    used.add(bp)
                if bp.reverse not in used:
                    result.append(bp.reverse)
                    used.add(bp.reverse)
        return result

    @cached_property
    def base_pair_graph(
        self,
    ) -> Dict[Residue3D, Set[Residue3D]]:
        graph = defaultdict(set)
        for pair in self.base_pairs:
            graph[pair.nt1_3d].add(pair.nt2_3d)
            graph[pair.nt2_3d].add(pair.nt1_3d)
        return graph

    @cached_property
    def base_pair_dict(self) -> Dict[Tuple[Residue3D, Residue3D], BasePair3D]:
        result = {}
        for base_pair in self.base_pairs:
            residue_i = base_pair.nt1_3d
            residue_j = base_pair.nt2_3d
            result[(residue_i, residue_j)] = base_pair
            result[(residue_j, residue_i)] = base_pair.reverse
        return result

    @cached_property
    def stackings(self) -> List[Stacking3D]:
        result = []
        used = set()
        for stacking in self.stackings2d:
            nt1 = self.structure3d.find_residue(stacking.nt1.label, stacking.nt1.auth)
            nt2 = self.structure3d.find_residue(stacking.nt2.label, stacking.nt2.auth)
            if nt1 is not None and nt2 is not None:
                st = Stacking3D(stacking.nt1, stacking.nt2, stacking.topology, nt1, nt2)
                if st not in used:
                    result.append(st)
                    used.add(st)
                if st.reverse not in used:
                    result.append(st.reverse)
                    used.add(st.reverse)
        return result

    @cached_property
    def stacking_graph(self) -> Dict[Residue3D, Set[Residue3D]]:
        graph = defaultdict(set)
        for pair in self.stackings:
            graph[pair.nt1_3d].add(pair.nt2_3d)
            graph[pair.nt2_3d].add(pair.nt1_3d)
        return graph

    @cached_property
    def strands_sequences(self) -> List[Tuple[str, str]]:
        nucleotides = [
            residue for residue in self.structure3d.residues if residue.is_nucleotide
        ]

        if len(nucleotides) == 0:
            return []

        result = []
        strand = [nucleotides[0]]

        for i in range(1, len(nucleotides)):
            previous = strand[-1]
            current = nucleotides[i]

            if previous.chain == current.chain and (
                self.find_gaps == False or previous.is_connected(current)
            ):
                strand.append(current)
            else:
                result.append(
                    (
                        previous.chain,
                        "".join([residue.one_letter_name for residue in strand]),
                    )
                )
                strand = [current]

        if len(strand) > 0:
            result.append(
                (
                    strand[0].chain,
                    "".join([residue.one_letter_name for residue in strand]),
                )
            )

        return result

    @cached_property
    def bpseq(self) -> BpSeq:
        return self.__generate_bpseq(
            [
                base_pair
                for base_pair in self.base_pairs
                if base_pair.is_canonical and base_pair.nt1 < base_pair.nt2
            ]
        )

    def __generate_bpseq(self, base_pairs):
        result: Dict[int, List] = {}
        residue_map: Dict[Residue3D, int] = {}
        i = 1
        for residue in self.structure3d.residues:
            if residue.is_nucleotide:
                result[i] = [i, residue.one_letter_name, 0]
                residue_map[residue] = i
                i += 1

        for base_pair in base_pairs:
            j = residue_map.get(base_pair.nt1_3d, None)
            k = residue_map.get(base_pair.nt2_3d, None)
            if j is None or k is None:
                continue
            result[j][2] = k
            result[k][2] = j

        return BpSeq(
            [
                Entry(index_, sequence, pair)
                for index_, sequence, pair in result.values()
            ]
        )

    @cached_property
    def dot_bracket(self) -> str:
        dbns = self.__generate_dot_bracket_per_strand(self.bpseq)
        i = 0
        result = []

        for i, pair in enumerate(self.strands_sequences):
            chain, sequence = pair
            result.append(f">strand_{chain}")
            result.append(sequence)
            result.append(dbns[i])
            i += len(sequence)
        return "\n".join(result)

    def __generate_dot_bracket_per_strand(self, bpseq: BpSeq) -> List[str]:
        dbn = bpseq.dot_bracket.structure
        i = 0
        result = []

        for _, sequence in self.strands_sequences:
            result.append("".join(dbn[i : i + len(sequence)]))
            i += len(sequence)
        return result

    @cached_property
    def extended_dot_bracket(self) -> str:
        result = [
            [f"    >strand_{chain}", f"seq {sequence}"]
            for chain, sequence in self.strands_sequences
        ]

        for lw in LeontisWesthof:
            row1, row2 = [], []
            used = set()

            for base_pair in self.base_pairs:
                if base_pair.lw == lw and base_pair.nt1 < base_pair.nt2:
                    if base_pair.nt1 not in used and base_pair.nt2 not in used:
                        row1.append(base_pair)
                        used.add(base_pair.nt1)
                        used.add(base_pair.nt2)
                    else:
                        row2.append(base_pair)

            for row in [row1, row2]:
                if row:
                    bpseq = self.__generate_bpseq(row)
                    dbns = self.__generate_dot_bracket_per_strand(bpseq)

                    for i in range(len(self.strands_sequences)):
                        result[i].append(f"{lw.value} {dbns[i]}")

        return "\n".join(["\n".join(r) for r in result])


def torsion_angle(a1: Atom, a2: Atom, a3: Atom, a4: Atom) -> float:
    v1 = a2.coordinates - a1.coordinates
    v2 = a3.coordinates - a2.coordinates
    v3 = a4.coordinates - a3.coordinates
    t1: numpy.typing.NDArray[numpy.floating] = numpy.cross(v1, v2)
    t2: numpy.typing.NDArray[numpy.floating] = numpy.cross(v2, v3)
    t3: numpy.typing.NDArray[numpy.floating] = v1 * numpy.linalg.norm(v2)
    return math.atan2(numpy.dot(t2, t3), numpy.dot(t1, t2))
