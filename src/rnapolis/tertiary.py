import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, total_ordering
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy
import numpy.typing
from scipy.stats import vonmises

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BpSeq,
    Entry,
    GlycosidicBond,
    InterStemParameters,
    LeontisWesthof,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Saenger,
    Stacking,
    Stem,
    Strand,
    Structure2D,
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
    entity_id: Optional[str]
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
    # Heavy atoms in phosphate and ribose
    phosphate_atoms = {"P", "OP1", "OP2", "O3'", "O5'"}
    sugar_atoms = {"C1'", "C2'", "C3'", "C4'", "C5'", "O4'"}
    # Heavy atoms for each main nucleobase
    nucleobase_heavy_atoms = {
        "A": set(["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"]),
        "G": set(["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"]),
        "C": set(["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]),
        "U": set(["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]),
        "T": set(["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C5M", "C6"]),
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
        scores = {"phosphate": 0.0, "sugar": 0.0, "base": 0.0, "connections": 0.0}
        weights = {"phosphate": 0.25, "sugar": 0.25, "base": 0.25, "connections": 0.25}

        residue_atoms = {atom.name for atom in self.atoms}

        phosphate_match = len(residue_atoms.intersection(self.phosphate_atoms))
        scores["phosphate"] = phosphate_match / len(self.phosphate_atoms)

        sugar_match = len(residue_atoms.intersection(self.sugar_atoms))
        scores["sugar"] = sugar_match / len(self.sugar_atoms)

        nucleobase_atoms = {
            key: self.nucleobase_heavy_atoms[key] for key in self.nucleobase_heavy_atoms
        }
        matches = {
            key: len(residue_atoms.intersection(nucleobase_atoms[key]))
            / len(nucleobase_atoms[key])
            for key in nucleobase_atoms
        }
        best_match = max(matches.items(), key=lambda x: x[1])
        scores["base"] = best_match[1]

        connection_score = 0.0
        distance_threshold = 2.0

        if "P" in residue_atoms and "O5'" in residue_atoms:
            p_atom = next(atom for atom in self.atoms if atom.name == "P")
            o5_atom = next(atom for atom in self.atoms if atom.name == "O5'")
            if (
                numpy.linalg.norm(p_atom.coordinates - o5_atom.coordinates)
                <= distance_threshold
            ):
                connection_score += 0.5
        if "C1'" in residue_atoms:
            c1_atom = next(atom for atom in self.atoms if atom.name == "C1'")
            for base_connection in ["N9", "N1"]:
                if base_connection in residue_atoms:
                    base_atom = next(
                        atom for atom in self.atoms if atom.name == base_connection
                    )
                    if (
                        numpy.linalg.norm(c1_atom.coordinates - base_atom.coordinates)
                        <= distance_threshold
                    ):
                        connection_score += 0.5
                        break

        scores["connections"] = connection_score

        probability = sum(
            scores[component] * weights[component] for component in scores.keys()
        )
        return probability > 0.5

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
        yield Atom(None, self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0, None)

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
        yield Atom(None, self.label, self.auth, self.model, "UNK", 0.0, 0.0, 0.0, None)


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

    def extract_secondary_structure(
        self, base_interactions: BaseInteractions, find_gaps: bool = False
    ) -> Tuple[Structure2D, "Mapping2D3D"]:
        """
        Create a secondary structure representation.

        Args:
            base_interactions: Interactions
            find_gaps: Whether to detect gaps in the structure
            all_dot_brackets: Whether to return all possible dot-bracket notations

        Returns:
            A tuple containing the Structure2D object, a list of dot-bracket notations,
            and the Mapping2D3D object.
        """
        mapping = Mapping2D3D(
            self,
            base_interactions.base_pairs,
            base_interactions.stackings,
            find_gaps,
        )
        stems, single_strands, hairpins, loops = mapping.bpseq.elements

        # Calculate inter-stem parameters using the helper function
        inter_stem_params = calculate_all_inter_stem_parameters(mapping)

        structure2d = Structure2D(
            base_interactions.base_pairs,
            base_interactions.stackings,
            base_interactions.base_ribose_interactions,
            base_interactions.base_phosphate_interactions,
            base_interactions.other_interactions,
            mapping.bpseq,
            mapping.bpseq_index_to_residue_map,
            mapping.dot_bracket,
            mapping.extended_dot_bracket,
            stems,
            single_strands,
            hairpins,
            loops,
            inter_stem_params,
        )
        return structure2d, mapping


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
        nucleotides = list(filter(lambda r: r.is_nucleotide, self.structure3d.residues))

        if not nucleotides:
            return []

        result = [(nucleotides[0].chain, [nucleotides[0].one_letter_name])]

        for i in range(1, len(nucleotides)):
            previous = nucleotides[i - 1]
            residue = nucleotides[i]

            if residue.chain != previous.chain:
                result.append((residue.chain, [residue.one_letter_name]))
            else:
                if self.find_gaps:
                    if not previous.is_connected(residue):
                        for k in range(residue.number - previous.number - 1):
                            result[-1][1].append("?")
                result[-1][1].append(residue.one_letter_name)

        return [(chain, "".join(sequence)) for chain, sequence in result]

    @cached_property
    def bpseq(self) -> BpSeq:
        return self._generated_bpseq_data[0]

    @cached_property
    def bpseq_index_to_residue_map(self) -> Dict[int, Residue3D]:
        """Mapping from BpSeq entry index to the corresponding Residue3D object."""
        return self._generated_bpseq_data[1]

    @cached_property
    def _generated_bpseq_data(self) -> Tuple[BpSeq, Dict[int, Residue3D]]:
        """Helper property to compute BpSeq and index map simultaneously."""

        def pair_scoring_function(pair: BasePair3D) -> int:
            if pair.saenger is not None:
                if pair.saenger in (Saenger.XIX, Saenger.XX):
                    return 0
                else:
                    return 1

            sequence = "".join(
                sorted(
                    [
                        pair.nt1_3d.one_letter_name.upper(),
                        pair.nt2_3d.one_letter_name.upper(),
                    ]
                )
            )
            if sequence in ("AU", "AT", "CG"):
                return 0
            return 1

        canonical = [
            base_pair
            for base_pair in self.base_pairs
            if base_pair.is_canonical and base_pair.nt1 < base_pair.nt2
        ]

        while True:
            matches = defaultdict(list)

            for base_pair in canonical:
                matches[base_pair.nt1_3d].append(base_pair)
                matches[base_pair.nt2_3d].append(base_pair)

            for pairs in matches.values():
                if len(pairs) > 1:
                    pairs = sorted(pairs, key=pair_scoring_function)
                    canonical.remove(pairs[-1])
                    break
            else:
                break

        return self.__generate_bpseq(canonical)

    def __generate_bpseq(self, base_pairs) -> Tuple[BpSeq, Dict[int, Residue3D]]:
        """Generates BpSeq entries and a map from index to Residue3D."""
        nucleotides = list(filter(lambda r: r.is_nucleotide, self.structure3d.residues))
        result: Dict[int, List] = {}
        residue_map: Dict[Residue3D, int] = {}
        index_to_residue_map: Dict[int, Residue3D] = {}
        i = 1

        for j, residue in enumerate(nucleotides):
            if self.find_gaps and j > 0:
                previous = nucleotides[j - 1]

                if (
                    not previous.is_connected(residue)
                    and previous.chain == residue.chain
                ):
                    for k in range(residue.number - previous.number - 1):
                        result[i] = [i, "?", 0]
                        i += 1

            result[i] = [i, residue.one_letter_name, 0]
            residue_map[residue] = i
            index_to_residue_map[i] = residue
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
        ), index_to_residue_map

    def find_residue_for_entry(self, entry: Entry) -> Optional[Residue3D]:
        """Finds the Residue3D object corresponding to a BpSeq Entry."""
        return self.bpseq_index_to_residue_map.get(entry.index_)

    def get_residues_for_strand(self, strand: Strand) -> List[Residue3D]:
        """Retrieves the list of Residue3D objects corresponding to a Strand."""
        residues = []
        # Strand indices are 1-based and inclusive
        for index_ in range(strand.first, strand.last + 1):
            residue = self.bpseq_index_to_residue_map.get(index_)
            if residue:
                residues.append(residue)
        return residues

    @cached_property
    def dot_bracket(self) -> str:
        dbns = self.__generate_dot_bracket_per_strand(self.bpseq.dot_bracket.structure)
        i = 0
        result = []

        for i, pair in enumerate(self.strands_sequences):
            chain, sequence = pair
            result.append(f">strand_{chain}")
            result.append(sequence)
            result.append(dbns[i])
            i += len(sequence)
        return "\n".join(result)

    def _calculate_pair_centroid(
        self, residue1: Residue3D, residue2: Residue3D
    ) -> Optional[numpy.typing.NDArray[numpy.floating]]:
        """Calculates the geometric mean of base atoms for a pair of residues."""
        base_atoms = []
        for residue in [residue1, residue2]:
            base_atom_names = Residue3D.nucleobase_heavy_atoms.get(
                residue.one_letter_name.upper(), set()
            )
            if not base_atom_names:
                logging.warning(
                    f"Could not find base atom definition for residue {residue.full_name}"
                )
                continue
            for atom in residue.atoms:
                if atom.name in base_atom_names:
                    base_atoms.append(atom)

        if not base_atoms:
            logging.warning(
                f"No base atoms found for pair {residue1.full_name} - {residue2.full_name}"
            )
            return None

        coordinates = [atom.coordinates for atom in base_atoms]
        return numpy.mean(coordinates, axis=0)

    def get_stem_coordinates(
        self, stem: Stem
    ) -> List[numpy.typing.NDArray[numpy.floating]]:
        """
        Calculates the geometric centroid for each base pair in the stem.

        Args:
            stem: The Stem object.

        Returns:
            A list of numpy arrays, where each array is the centroid of a
            base pair in the stem. Returns an empty list if no centroids
            can be calculated.
        """
        all_pair_centroids = []
        stem_len = stem.strand5p.last - stem.strand5p.first + 1

        for i in range(stem_len):
            idx5p = stem.strand5p.first + i
            idx3p = stem.strand3p.last - i
            try:
                res5p = self.bpseq_index_to_residue_map[idx5p]
                res3p = self.bpseq_index_to_residue_map[idx3p]
                centroid = self._calculate_pair_centroid(res5p, res3p)
                if centroid is not None:
                    all_pair_centroids.append(centroid)
            except KeyError:
                logging.warning(
                    f"Could not find residues for pair {idx5p}-{idx3p} in stem {stem}"
                )
                continue  # Continue calculating other centroids

        return all_pair_centroids

    def calculate_inter_stem_parameters(
        self, stem1: Stem, stem2: Stem, kappa: float = 10.0
    ) -> Optional[Dict[str, Union[str, float]]]:
        """
        Calculates geometric parameters between two stems based on closest endpoints
        and the probability of the observed torsion angle based on an expected
        A-RNA twist using a von Mises distribution.

        Args:
            stem1: The first Stem object.
            stem2: The second Stem object.
            kappa: Concentration parameter for the von Mises distribution (default: 10.0).

        Returns:
            A dictionary containing:
            - 'type': The type of closest endpoint pair ('cs55', 'cs53', 'cs35', 'cs33').
            - 'torsion_angle': The calculated torsion angle in degrees.
            - 'min_endpoint_distance': The minimum distance between the endpoints.
            - 'torsion_angle_pdf': The probability density function (PDF) value of the
              torsion angle under the von Mises distribution.
            - 'min_endpoint_distance_pdf': The probability density function (PDF) value
              based on the minimum endpoint distance using a Lennard-Jones-like function.
            - 'coaxial_probability': The normalized product of the torsion angle PDF and
              distance PDF, indicating the likelihood of coaxial stacking (0-1).
            Returns None if either stem has fewer than 2 base pairs or centroids
            cannot be calculated.
        """
        stem1_centroids = self.get_stem_coordinates(stem1)
        stem2_centroids = self.get_stem_coordinates(stem2)

        # Need at least 2 centroids (base pairs) per stem
        if len(stem1_centroids) < 2 or len(stem2_centroids) < 2:
            logging.warning(
                f"Cannot calculate inter-stem parameters for stems {stem1} and {stem2}: "
                f"Insufficient base pairs ({len(stem1_centroids)} and {len(stem2_centroids)} respectively)."
            )
            return None

        # Define the endpoints for each stem
        s1_first, s1_last = stem1_centroids[0], stem1_centroids[-1]
        s2_first, s2_last = stem2_centroids[0], stem2_centroids[-1]

        # Calculate distances between the four endpoint pairs
        endpoint_distances = {
            "cs55": numpy.linalg.norm(s1_first - s2_first),
            "cs53": numpy.linalg.norm(s1_first - s2_last),
            "cs35": numpy.linalg.norm(s1_last - s2_first),
            "cs33": numpy.linalg.norm(s1_last - s2_last),
        }

        # Find the minimum endpoint distance and the corresponding pair
        min_endpoint_distance = min(endpoint_distances.values())
        closest_pair_key = min(endpoint_distances, key=endpoint_distances.get)

        # Select the points for torsion and determine mu based on the closest pair.
        # s1p2 and s2p1 must be the endpoints involved in the minimum distance.
        a_rna_twist = 32.7
        mu_degrees = 0.0

        if closest_pair_key == "cs55":
            # Closest: s1_first and s2_first
            # Torsion points: s1_second, s1_first, s2_first, s2_second
            s1p1, s1p2 = stem1_centroids[1], stem1_centroids[0]
            s2p1, s2p2 = stem2_centroids[0], stem2_centroids[1]
            mu_degrees = 180.0 - a_rna_twist
        elif closest_pair_key == "cs53":
            # Closest: s1_first and s2_last
            # Torsion points: s1_second, s1_first, s2_last, s2_second_last
            s1p1, s1p2 = stem1_centroids[1], stem1_centroids[0]
            s2p1, s2p2 = stem2_centroids[-1], stem2_centroids[-2]
            mu_degrees = 0.0 - a_rna_twist
        elif closest_pair_key == "cs35":
            # Closest: s1_last and s2_first
            # Torsion points: s1_second_last, s1_last, s2_first, s2_second
            s1p1, s1p2 = stem1_centroids[-2], stem1_centroids[-1]
            s2p1, s2p2 = stem2_centroids[0], stem2_centroids[1]
            mu_degrees = 0.0 + a_rna_twist
        elif closest_pair_key == "cs33":
            # Closest: s1_last and s2_last
            # Torsion points: s1_second_last, s1_last, s2_last, s2_second_last
            s1p1, s1p2 = stem1_centroids[-2], stem1_centroids[-1]
            s2p1, s2p2 = stem2_centroids[-1], stem2_centroids[-2]
            mu_degrees = 180.0 + a_rna_twist
        else:
            # This case should ideally not be reached if endpoint_distances is not empty
            logging.error(
                f"Unexpected closest pair key: {closest_pair_key}. Cannot calculate parameters."
            )
            return None

        # Calculate torsion angle (in radians)
        torsion_radians = calculate_torsion_angle_coords(s1p1, s1p2, s2p1, s2p2)

        # Create von Mises distribution instance
        mu_radians = math.radians(mu_degrees)
        vm_dist = vonmises(kappa=kappa, loc=mu_radians)

        # Calculate the probability density function (PDF) value for the torsion angle
        torsion_probability = vm_dist.pdf(torsion_radians)

        # Calculate the probability density for the minimum endpoint distance
        distance_probability = distance_pdf(
            min_endpoint_distance
        )  # Use the new function

        # Calculate the coaxial probability
        # Max torsion probability occurs at mu (location of the distribution)
        max_torsion_probability = vm_dist.pdf(mu_radians)
        # Max distance probability is 1.0 by design of lennard_jones_like_pdf
        max_distance_probability = 1.0
        # Normalization factor is the product of maximum possible probabilities
        normalization_factor = max_torsion_probability * max_distance_probability

        coaxial_probability = 0.0
        if normalization_factor > 1e-9:  # Avoid division by zero
            probability_product = torsion_probability * distance_probability
            coaxial_probability = probability_product / normalization_factor
            # Clamp between 0 and 1
            coaxial_probability = max(0.0, min(1.0, coaxial_probability))

        return {
            "type": closest_pair_key,
            "torsion_angle": math.degrees(torsion_radians),
            "min_endpoint_distance": min_endpoint_distance,
            "torsion_angle_pdf": torsion_probability,
            "min_endpoint_distance_pdf": distance_probability,
            "coaxial_probability": coaxial_probability,
        }

    def __generate_dot_bracket_per_strand(self, dbn_structure: str) -> List[str]:
        dbn = dbn_structure
        i = 0
        result = []

        for _, sequence in self.strands_sequences:
            result.append("".join(dbn[i : i + len(sequence)]))
            i += len(sequence)
        return result

    @cached_property
    def all_dot_brackets(self) -> List[str]:
        dot_brackets = []

        for dot_bracket in self.bpseq.all_dot_brackets:
            dbns = self.__generate_dot_bracket_per_strand(dot_bracket.structure)
            i = 0
            result = []

            for i, pair in enumerate(self.strands_sequences):
                chain, sequence = pair
                result.append(f">strand_{chain}")
                result.append(sequence)
                result.append(dbns[i])
                i += len(sequence)
            dot_brackets.append("\n".join(result))

        return dot_brackets

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
                    bpseq, _ = self.__generate_bpseq(row)  # Unpack the tuple
                    dbns = self.__generate_dot_bracket_per_strand(
                        bpseq.dot_bracket.structure
                    )

                    for i in range(len(self.strands_sequences)):
                        result[i].append(f"{lw.value} {dbns[i]}")

        return "\n".join(["\n".join(r) for r in result])


def distance_pdf(
    x: float, lower_bound: float = 3.0, upper_bound: float = 7.0, steepness: float = 5.0
) -> float:
    """
    Calculates a probability density based on distance using a plateau function.

    The function uses the product of two sigmoid functions to create a distribution
    that is close to 1.0 between lower_bound and upper_bound, and drops off
    rapidly outside this range.

    Args:
        x: The distance value.
        lower_bound: The start of the high-probability plateau (default: 3.0).
        upper_bound: The end of the high-probability plateau (default: 7.0).
        steepness: Controls how quickly the probability drops outside the plateau
                   (default: 5.0). Higher values mean steeper drops.

    Returns:
        The calculated probability density (between 0.0 and 1.0).
    """
    # Define a maximum exponent value to prevent overflow
    max_exponent = 700.0

    # Calculate exponent for the first sigmoid (increasing)
    exponent1 = -steepness * (x - lower_bound)
    # Clamp the exponent if it's excessively large (which happens when x << lower_bound)
    exponent1 = min(exponent1, max_exponent)
    sigmoid1 = 1.0 / (1.0 + math.exp(exponent1))

    # Calculate exponent for the second sigmoid (decreasing)
    exponent2 = steepness * (x - upper_bound)
    # Clamp the exponent if it's excessively large (which happens when x >> upper_bound)
    exponent2 = min(exponent2, max_exponent)
    sigmoid2 = 1.0 / (1.0 + math.exp(exponent2))

    # The product creates the plateau effect
    probability = sigmoid1 * sigmoid2
    # Clamp to handle potential floating point inaccuracies near 0 and 1
    return max(0.0, min(1.0, probability))


def calculate_all_inter_stem_parameters(
    mapping: Mapping2D3D,
) -> List[InterStemParameters]:
    """
    Calculates InterStemParameters for all valid pairs of stems found in the mapping.

    Args:
        mapping: The Mapping2D3D object containing structure, 2D info, and mapping.

    """
    stems = mapping.bpseq.elements[0]  # Get stems from mapping
    inter_stem_params = []
    for i, j in itertools.combinations(range(len(stems)), 2):
        stem1 = stems[i]
        stem2 = stems[j]

        # Ensure both stems have at least 2 base pairs for parameter calculation
        if (stem1.strand5p.last - stem1.strand5p.first + 1) > 1 and (
            stem2.strand5p.last - stem2.strand5p.first + 1
        ) > 1:
            params = mapping.calculate_inter_stem_parameters(stem1, stem2)
            # Only add if calculation returned valid values
            if params is not None:
                inter_stem_params.append(
                    InterStemParameters(
                        stem1_idx=i,
                        stem2_idx=j,
                        type=params["type"],
                        torsion=params["torsion_angle"],
                        min_endpoint_distance=params["min_endpoint_distance"],
                        torsion_angle_pdf=params["torsion_angle_pdf"],
                        min_endpoint_distance_pdf=params["min_endpoint_distance_pdf"],
                        coaxial_probability=params["coaxial_probability"],
                    )
                )
    return inter_stem_params


def torsion_angle(a1: Atom, a2: Atom, a3: Atom, a4: Atom) -> float:
    """Calculates the torsion angle between four atoms."""
    return calculate_torsion_angle_coords(
        a1.coordinates, a2.coordinates, a3.coordinates, a4.coordinates
    )


def calculate_torsion_angle_coords(
    p1: numpy.typing.NDArray[numpy.floating],
    p2: numpy.typing.NDArray[numpy.floating],
    p3: numpy.typing.NDArray[numpy.floating],
    p4: numpy.typing.NDArray[numpy.floating],
) -> float:
    """Calculates the torsion angle between four points defined by their coordinates."""
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3

    # Normalize vectors to avoid issues with very short vectors
    v1_norm = v1 / numpy.linalg.norm(v1) if numpy.linalg.norm(v1) > 1e-6 else v1
    v2_norm = v2 / numpy.linalg.norm(v2) if numpy.linalg.norm(v2) > 1e-6 else v2
    v3_norm = v3 / numpy.linalg.norm(v3) if numpy.linalg.norm(v3) > 1e-6 else v3

    t1 = numpy.cross(v1_norm, v2_norm)
    t2 = numpy.cross(v2_norm, v3_norm)
    t3 = v1_norm * numpy.linalg.norm(v2_norm)

    # Ensure t1 and t2 are not zero vectors before calculating dot products
    if numpy.linalg.norm(t1) < 1e-6 or numpy.linalg.norm(t2) < 1e-6:
        return 0.0  # Or handle as undefined/error

    dot_t1_t2 = numpy.dot(t1, t2)
    dot_t2_t3 = numpy.dot(t2, t3)

    # Clamp dot product arguments for acos/atan2 to avoid domain errors
    dot_t1_t2 = numpy.clip(dot_t1_t2, -1.0, 1.0)

    angle = math.atan2(dot_t2_t3, dot_t1_t2)
    return angle if not math.isnan(angle) else 0.0
