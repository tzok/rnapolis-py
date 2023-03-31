import logging
import string
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property, total_ordering
from typing import Dict, List, Optional, Tuple


class Molecule(Enum):
    DNA = "DNA"
    RNA = "RNA"
    Other = "Other"


class GlycosidicBond(Enum):
    anti = "anti"
    syn = "syn"


@total_ordering
class LeontisWesthof(Enum):
    cWW = "cWW"
    cWH = "cWH"
    cWS = "cWS"
    cHW = "cHW"
    cHH = "cHH"
    cHS = "cHS"
    cSW = "cSW"
    cSH = "cSH"
    cSS = "cSS"
    tWW = "tWW"
    tWH = "tWH"
    tWS = "tWS"
    tHW = "tHW"
    tHH = "tHH"
    tHS = "tHS"
    tSW = "tSW"
    tSH = "tSH"
    tSS = "tSS"

    @property
    def reverse(self):
        return LeontisWesthof[f"{self.name[0]}{self.name[2]}{self.name[1]}"]

    def __lt__(self, other):
        return tuple(self.value) < tuple(other.value)


class Saenger(Enum):
    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    V = "V"
    VI = "VI"
    VII = "VII"
    VIII = "VIII"
    IX = "IX"
    X = "X"
    XI = "XI"
    XII = "XII"
    XIII = "XIII"
    XIV = "XIV"
    XV = "XV"
    XVI = "XVI"
    XVII = "XVII"
    XVIII = "XVIII"
    XIX = "XIX"
    XX = "XX"
    XXI = "XXI"
    XXII = "XXII"
    XXIII = "XXIII"
    XXIV = "XXIV"
    XXV = "XXV"
    XXVI = "XXVI"
    XXVII = "XXVII"
    XXVIII = "XXVIII"

    @staticmethod
    def table() -> Dict[Tuple[str, str], str]:
        return {
            ("AA", "tWW"): "I",
            ("AA", "tHH"): "II",
            ("GG", "tWW"): "III",
            ("GG", "tSS"): "IV",
            ("AA", "tWH"): "V",
            ("AA", "tHW"): "V",
            ("GG", "cWH"): "VI",
            ("GG", "cHW"): "VI",
            ("GG", "tWH"): "VII",
            ("GG", "tHW"): "VII",
            ("AG", "cWW"): "VIII",
            ("GA", "cWW"): "VIII",
            ("AG", "cHW"): "IX",
            ("GA", "cWH"): "IX",
            ("AG", "tWS"): "X",
            ("GA", "tSW"): "X",
            ("AG", "tHS"): "XI",
            ("GA", "tSH"): "XI",
            ("UU", "tWW"): "XII",
            ("TT", "tWW"): "XII",
            # XIII is UU/TT in tWW but donor-donor, so impossible
            # XIV and XV are both CC in tWW but donor-donor, so impossible
            ("UU", "cWW"): "XVI",
            ("TT", "cWW"): "XVI",
            ("CU", "tWW"): "XVII",
            ("UC", "tWW"): "XVII",
            ("CU", "cWW"): "XVIII",
            ("UC", "cWW"): "XVIII",
            ("CG", "cWW"): "XIX",
            ("GC", "cWW"): "XIX",
            ("AU", "cWW"): "XX",
            ("UA", "cWW"): "XX",
            ("AT", "cWW"): "XX",
            ("TA", "cWW"): "XX",
            ("AU", "tWW"): "XXI",
            ("UA", "tWW"): "XXI",
            ("AT", "tWW"): "XXI",
            ("TA", "tWW"): "XXI",
            ("CG", "tWW"): "XXII",
            ("GC", "tWW"): "XXII",
            ("AU", "cHW"): "XXIII",
            ("UA", "cWH"): "XXIII",
            ("AT", "cHW"): "XXIII",
            ("TA", "cWH"): "XXIII",
            ("AU", "tHW"): "XXIV",
            ("UA", "tWH"): "XXIV",
            ("AT", "tHW"): "XXIV",
            ("TA", "tWH"): "XXIV",
            ("AC", "tHW"): "XXV",
            ("CA", "tWH"): "XXV",
            ("AC", "tWW"): "XXVI",
            ("CA", "tWW"): "XXVI",
            ("GU", "tWW"): "XXVII",
            ("UG", "tWW"): "XXVII",
            ("GT", "tWW"): "XXVII",
            ("TG", "tWW"): "XXVII",
            ("GU", "cWW"): "XXVIII",
            ("UG", "cWW"): "XXVIII",
            ("GT", "cWW"): "XXVIII",
            ("TG", "cWW"): "XXVIII",
        }

    @property
    def is_canonical(self) -> bool:
        return self == Saenger.XIX or self == Saenger.XX or self == Saenger.XXVIII


class StackingTopology(Enum):
    upward = "upward"
    downward = "downward"
    inward = "inward"
    outward = "outward"

    @property
    def reverse(self):
        if self == StackingTopology.upward:
            return StackingTopology.downward
        elif self == StackingTopology.downward:
            return StackingTopology.upward
        return self


class BR(Enum):
    _0 = "0BR"
    _1 = "1BR"
    _2 = "2BR"
    _3 = "3BR"
    _4 = "4BR"
    _5 = "5BR"
    _6 = "6BR"
    _7 = "7BR"
    _8 = "8BR"
    _9 = "9BR"


class BPh(Enum):
    _0 = "0BPh"
    _1 = "1BPh"
    _2 = "2BPh"
    _3 = "3BPh"
    _4 = "4BPh"
    _5 = "5BPh"
    _6 = "6BPh"
    _7 = "7BPh"
    _8 = "8BPh"
    _9 = "9BPh"


@dataclass(frozen=True, order=True)
class ResidueLabel:
    chain: str
    number: int
    name: str


@dataclass(frozen=True, order=True)
class ResidueAuth:
    chain: str
    number: int
    icode: Optional[str]
    name: str


@dataclass(frozen=True)
@total_ordering
class Residue:
    label: Optional[ResidueLabel]
    auth: Optional[ResidueAuth]

    def __lt__(self, other):
        return (self.chain, self.number, self.icode or " ") < (
            other.chain,
            other.number,
            other.icode or " ",
        )

    @property
    def chain(self) -> Optional[str]:
        if self.auth is not None:
            return self.auth.chain
        if self.label is not None:
            return self.label.chain
        return None

    @property
    def number(self) -> Optional[int]:
        if self.auth is not None:
            return self.auth.number
        if self.label is not None:
            return self.label.number
        return None

    @property
    def icode(self) -> Optional[str]:
        if self.auth is not None:
            return self.auth.icode if self.auth.icode not in (" ", "?") else None
        return None

    @property
    def name(self) -> Optional[str]:
        if self.auth is not None:
            return self.auth.name
        if self.label is not None:
            return self.label.name
        return None

    @property
    def molecule_type(self) -> Molecule:
        if self.name is not None:
            if self.name.upper() in ("A", "C", "G", "U"):
                return Molecule.RNA
            if self.name.upper() in ("DA", "DC", "DG", "DT"):
                return Molecule.DNA
        return Molecule.Other

    @property
    @cache
    def full_name(self) -> Optional[str]:
        if self.auth is not None:
            if self.auth.chain.isspace():
                builder = f"{self.auth.name}"
            else:
                builder = f"{self.auth.chain}.{self.auth.name}"
            if len(self.auth.name) > 0 and self.auth.name[-1] in string.digits:
                builder += "/"
            builder += f"{self.auth.number}"
            if self.auth.icode:
                builder += f"^{self.auth.icode}"
            return builder
        elif self.label is not None:
            if self.label.chain.isspace():
                builder = f"{self.label.name}"
            else:
                builder = f"{self.label.chain}.{self.label.name}"
            if len(self.label.name) > 0 and self.label.name[-1] in string.digits:
                builder += "/"
            builder += f"{self.label.number}"
            return builder
        return None


@dataclass(frozen=True, order=True)
class Interaction:
    nt1: Residue
    nt2: Residue


@dataclass(frozen=True, order=True)
class BasePair(Interaction):
    lw: LeontisWesthof
    saenger: Optional[Saenger]


@dataclass(frozen=True, order=True)
class Stacking(Interaction):
    topology: Optional[StackingTopology]


@dataclass(frozen=True, order=True)
class BaseRibose(Interaction):
    br: Optional[BR]


@dataclass(frozen=True, order=True)
class BasePhosphate(Interaction):
    bph: Optional[BPh]


@dataclass(frozen=True, order=True)
class OtherInteraction(Interaction):
    pass


@dataclass(frozen=True, order=True)
class Structure2D:
    basePairs: List[BasePair]
    stackings: List[Stacking]
    baseRiboseInteractions: List[BaseRibose]
    basePhosphateInteractions: List[BasePhosphate]
    otherInteractions: List[OtherInteraction]


@dataclass
class Entry(Sequence):
    index_: int
    sequence: str
    pair: int

    def __getitem__(self, item):
        if item == 0:
            return self.index_
        elif item == 1:
            return self.sequence
        elif item == 2:
            return self.pair
        raise IndexError()

    def __len__(self) -> int:
        return 3

    def __str__(self):
        return f"{self.index_} {self.sequence} {self.pair}"


@dataclass(frozen=True)
class Strand:
    first: int
    last: int
    sequence: str
    structure: str

    @staticmethod
    def from_bpseq_entries(
        entries: List[Entry], dotbracket: str, reverse: bool = False
    ):
        first = entries[0].index_
        last = first + len(entries) - 1
        if reverse:
            first, last = last, first
        sequence = "".join(
            [
                entry.sequence
                for entry in (entries if not reverse else reversed(entries))
            ]
        )
        structure = dotbracket[first - 1 : last]
        return Strand(first, last, sequence, structure)

    def __str__(self):
        return f"{self.first}-{self.sequence}-{self.last}"


@dataclass(frozen=True)
class SingleStrand:
    strand: Strand
    is5p: bool
    is3p: bool

    def __str__(self):
        if self.is5p:
            return f"SingleStrand5p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        if self.is3p:
            return f"SingleStrand3p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        return f"SingleStrand {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass(frozen=True)
class Stem:
    strand5p: Strand
    strand3p: Strand

    @staticmethod
    def from_bpseq_entries(
        strand5p_entries: List[Entry], all_entries: List, dotbracket: str
    ):
        paired = set([entry[2] for entry in strand5p_entries])
        strand3p_entries = list(filter(lambda entry: entry[0] in paired, all_entries))
        return Stem(
            Strand.from_bpseq_entries(strand5p_entries, dotbracket),
            Strand.from_bpseq_entries(strand3p_entries, dotbracket),
        )

    def __str__(self):
        return f"Stem {self.strand5p.first} {self.strand5p.last} {self.strand5p.sequence} {self.strand5p.structure} {self.strand3p.first} {self.strand3p.last} {self.strand3p.sequence} {self.strand3p.structure}"


@dataclass(frozen=True)
class Hairpin:
    strand: Strand

    def __str__(self):
        return f"Hairpin {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass(frozen=True)
class Loop:
    strands: List[Strand]

    def __str__(self):
        desc = " ".join(
            [
                "{} {} {} {}".format(
                    strand.first, strand.last, strand.sequence, strand.structure
                )
                for strand in self.strands
            ]
        )
        return f"Loop {desc}"


@dataclass
class BpSeq:
    entries: List[Entry]

    @staticmethod
    def from_file(bpseq_path: str):
        entries = []
        with open(bpseq_path) as fd:
            for line in fd:
                fields = line.strip().split()
                if len(fields) != 3:
                    logging.warning("Failed to find 3 columns in BpSeq line: {}", line)
                    continue
                entry = Entry(int(fields[0]), fields[1], int(fields[2]))
                entries.append(entry)
        return BpSeq(entries)

    @staticmethod
    def from_dotbracket(dot_bracket):
        entries = [
            Entry(i + 1, dot_bracket.sequence[i], 0)
            for i in range(len(dot_bracket.sequence))
        ]
        for i, j in dot_bracket.pairs:
            entries[i].pair = j + 1
            entries[j].pair = i + 1
        return BpSeq(entries)

    def __post_init__(self):
        self.pairs = {}
        for i, _, j in self.entries:
            if j != 0:
                self.pairs[i] = j
                self.pairs[j] = i

    def __str__(self):
        return "\n".join(("{} {} {}".format(i, c, j) for i, c, j in self.entries))

    def __eq__(self, other):
        return len(self.entries) == len(other.entries) and all(
            ei == ej for ei, ej in zip(self.entries, other.entries)
        )

    @cached_property
    def sequence(self) -> str:
        return "".join(entry.sequence for entry in self.entries)

    def paired(self, only5to3: bool = False):
        result = filter(lambda entry: entry.pair != 0, self.entries)
        if only5to3:
            result = filter(lambda entry: entry.index_ < entry.pair, result)
        return result

    @cached_property
    def __stems(self) -> List:
        result = []
        strand5p: List[Entry] = []

        for entry in self.paired(only5to3=True):
            if not strand5p:
                strand5p.append(entry)
                continue

            i, _, j = entry
            k, _, l = strand5p[-1]
            if i == k + 1 and j == l - 1:
                strand5p.append(entry)
                continue

            result.append(
                Stem.from_bpseq_entries(strand5p, self.entries, self.fcfs.structure)
            )
            strand5p = [entry]

        if strand5p:
            result.append(
                Stem.from_bpseq_entries(strand5p, self.entries, self.fcfs.structure)
            )

        return result

    @cached_property
    def elements(self):
        if not self.__stems:
            return []

        stops = set()

        for stem in self.__stems:
            stops.add(stem.strand5p.first - 1)
            stops.add(stem.strand5p.last - 1)
            stops.add(stem.strand3p.first - 1)
            stops.add(stem.strand3p.last - 1)

        stops = sorted(stops)
        elements = [stem for stem in self.__stems]
        loop_candidates = []

        # 5' single strand
        if stops[0] > 0:
            elements.append(
                SingleStrand(
                    Strand.from_bpseq_entries(
                        self.entries[: stops[0] + 1], self.fcfs.structure
                    ),
                    True,
                    False,
                )
            )

        # single strands
        for i in range(1, len(stops)):
            candidate = self.entries[stops[i - 1] : stops[i] + 1]
            if all([entry.pair == 0 for entry in candidate[1:-1]]):
                if candidate[0].pair == candidate[-1].index_:
                    elements.append(
                        Hairpin(
                            Strand.from_bpseq_entries(candidate, self.fcfs.structure)
                        )
                    )
                else:
                    loop_candidates.append(
                        Strand.from_bpseq_entries(candidate, self.fcfs.structure)
                    )

        # 3' single strand
        if stops[-1] < len(self.entries) - 1:
            elements.append(
                SingleStrand(
                    Strand.from_bpseq_entries(
                        self.entries[stops[-1] :], self.fcfs.structure
                    ),
                    False,
                    True,
                )
            )

        graph = defaultdict(set)

        for i in range(len(loop_candidates)):
            for j in range(i + 1, len(loop_candidates)):
                i_first, i_last = loop_candidates[i].first, loop_candidates[i].last
                j_first, j_last = loop_candidates[j].first, loop_candidates[j].last
                if self.entries[i_last - 1].pair == j_first:
                    graph[i].add(j)
                if self.entries[j_last - 1].pair == i_first:
                    graph[j].add(i)

        used = set()

        for i in range(len(loop_candidates)):
            if i in used:
                continue

            loop = [loop_candidates[i]]
            used.add(i)

            while True:
                for j in graph[i]:
                    if j not in used:
                        loop.append(loop_candidates[j])
                        used.add(j)
                        i = j
                        break
                else:
                    break

            if self.entries[loop[0].first - 1].pair == loop[-1].last:
                if not all([strand.last - strand.first <= 1 for strand in loop]):
                    elements.append(Loop(loop))

        for i in range(len(loop_candidates)):
            if i not in used:
                elements.append(SingleStrand(loop_candidates[i]))

        return elements

    @cached_property
    def fcfs(self):
        stems = [(entry.index_, entry.pair, 1) for entry in self.paired(only5to3=True)]
        orders = [0 for i in range(len(stems))]

        for i in range(1, len(stems)):
            k, l, _ = stems[i]
            available = [True for i in range(10)]

            for j in range(i):
                m, n, _ = stems[j]
                conflicted = (k < m < l < n) or (m < k < n < l)

                if conflicted:
                    available[orders[j]] = False

            order = next(filter(lambda i: available[i] is True, range(len(available))))
            orders[i] = order

        sequence = self.sequence
        structure = ["." for _ in range(len(sequence))]
        brackets = ["()", "[]", "{}", "<>", "Aa", "Bb", "Cc"]

        for i, stem in enumerate(stems):
            bracket = brackets[orders[i]]
            j, k, n = stem

            while n > 0:
                structure[j - 1] = bracket[0]
                structure[k - 1] = bracket[1]
                j += 1
                k -= 1
                n -= 1

        structure = "".join(structure)

        return DotBracket.from_string(sequence, structure)


@dataclass
class DotBracket:
    sequence: str
    structure: str

    @staticmethod
    def from_file(path: str):
        with open(path) as f:
            lines = f.readlines()
        if len(lines) == 2:
            return DotBracket.from_string(lines[0].rstrip(), lines[1].rstrip())
        if len(lines) == 3:
            return DotBracket.from_string(lines[1].rstrip(), lines[2].rstrip())
        raise RuntimeError(f"Failed to read DotBracket from file: {path}")

    @staticmethod
    def from_string(sequence: str, structure: str):
        if len(sequence) != len(structure):
            raise ValueError(
                "Sequence and structure lengths differ, {} vs {}",
                (len(sequence), len(structure)),
            )
        return DotBracket(sequence, structure)

    def __post_init__(self):
        self.pairs = []

        opening = "([{<ABC"
        closing = ")]}>abc"
        begins = {bracket: list() for bracket in opening}
        matches = {end: begin for begin, end in zip(opening, closing)}

        for i in range(len(self.structure)):
            c = self.structure[i]
            if c in opening:
                begins[c].append(i)
            elif c in closing:
                begin = matches[c]
                self.pairs.append((begins[begin].pop(), i))

    def __str__(self):
        return f"{self.sequence}\n{self.structure}"
