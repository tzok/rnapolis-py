import itertools
import logging
import os
import re
import string
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from functools import cache, cached_property, total_ordering
from typing import Dict, List, Optional, Tuple

import graphviz
import pulp

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)


class Molecule(Enum):
    """Simple classification of molecule type."""

    DNA = "DNA"
    RNA = "RNA"
    Other = "Other"


class GlycosidicBond(Enum):
    """Orientation of the glycosidic bond."""

    anti = "anti"
    syn = "syn"


@total_ordering
class LeontisWesthof(Enum):
    """Leontis–Westhof base pair geometry classification."""

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
        """Return the Leontis–Westhof class with swapped edges."""
        return LeontisWesthof[f"{self.name[0]}{self.name[2]}{self.name[1]}"]

    def __lt__(self, other):
        """Compare Leontis–Westhof classes using tuple ordering of values."""
        return tuple(self.value) < tuple(other.value)


class Saenger(Enum):
    """Saenger base pair classification."""

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
        """Return mapping from (base pair, Leontis–Westhof) to Saenger class name."""
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

    @classmethod
    def from_leontis_westhof(
        cls,
        residue_i_one_letter_name: str,
        residue_j_one_letter_name: str,
        lw: LeontisWesthof,
    ) -> Optional["Saenger"]:
        """Map a Leontis–Westhof class and base pair to a Saenger class.

        Args:
            residue_i_one_letter_name: One-letter code of the first base.
            residue_j_one_letter_name: One-letter code of the second base.
            lw: Leontis–Westhof classification.

        Returns:
            Matching Saenger class or ``None`` if not defined.
        """
        key = (f"{residue_i_one_letter_name}{residue_j_one_letter_name}", lw.value)
        if key in Saenger.table():
            return Saenger[Saenger.table()[key]]
        return None

    @property
    def is_canonical(self) -> bool:
        """Return True for canonical Watson–Crick or wobble pairs."""
        return self == Saenger.XIX or self == Saenger.XX or self == Saenger.XXVIII


class StackingTopology(Enum):
    """Relative orientation of stacked bases."""

    upward = "upward"
    downward = "downward"
    inward = "inward"
    outward = "outward"

    @property
    def reverse(self):
        """Return stacking topology with reversed direction."""
        if self == StackingTopology.upward:
            return StackingTopology.downward
        elif self == StackingTopology.downward:
            return StackingTopology.upward
        return self


class BR(Enum):
    """Base–ribose interaction classes."""

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
    """Base–phosphate interaction classes."""

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
    """Label-style residue identifier (label chain/number/name)."""

    chain: str
    number: int
    name: str


@dataclass(frozen=True, order=True)
class ResidueAuth:
    """Auth-style residue identifier (auth chain/number/icode/name)."""

    chain: str
    number: int
    icode: Optional[str]
    name: str


@dataclass(frozen=True)
@total_ordering
class Residue:
    """Unified residue identifier with both label and auth coordinates."""

    label: Optional[ResidueLabel]
    auth: Optional[ResidueAuth]

    def __lt__(self, other):
        """Compare residues by chain, number and insertion code."""
        return (self.chain, self.number, self.icode or " ") < (
            other.chain,
            other.number,
            other.icode or " ",
        )

    @property
    def chain(self) -> Optional[str]:
        """Return chain identifier from auth or label coordinates."""
        if self.auth is not None:
            return self.auth.chain
        if self.label is not None:
            return self.label.chain
        return None

    @property
    def number(self) -> Optional[int]:
        """Return residue number from auth or label coordinates."""
        if self.auth is not None:
            return self.auth.number
        if self.label is not None:
            return self.label.number
        return None

    @property
    def icode(self) -> Optional[str]:
        """Return insertion code or ``None`` if not set or blank."""
        if self.auth is not None:
            return self.auth.icode if self.auth.icode not in (" ", "?") else None
        return None

    @property
    def name(self) -> Optional[str]:
        """Return residue name from auth or label coordinates."""
        if self.auth is not None:
            return self.auth.name
        if self.label is not None:
            return self.label.name
        return None

    @property
    def molecule_type(self) -> Molecule:
        """Classify residue as RNA, DNA or Other based on its name."""
        if self.name is not None:
            if self.name.upper() in ("A", "C", "G", "U"):
                return Molecule.RNA
            if self.name.upper() in ("DA", "DC", "DG", "DT"):
                return Molecule.DNA
        return Molecule.Other

    @property
    @cache
    def full_name(self) -> Optional[str]:
        """Return human-readable residue identifier (e.g. A.A/23^A)."""
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
    """Base class for all interactions between two residues."""

    nt1: Residue
    nt2: Residue


@dataclass(frozen=True, order=True)
class BasePair(Interaction):
    """Base pair interaction with Leontis–Westhof and Saenger class."""

    lw: LeontisWesthof
    saenger: Optional[Saenger]


@dataclass(frozen=True, order=True)
class Stacking(Interaction):
    """Base stacking interaction."""

    topology: Optional[StackingTopology]


@dataclass(frozen=True, order=True)
class BaseRibose(Interaction):
    """Base–ribose interaction."""

    br: Optional[BR]


@dataclass(frozen=True, order=True)
class BasePhosphate(Interaction):
    """Base–phosphate interaction."""

    bph: Optional[BPh]


@dataclass(frozen=True, order=True)
class OtherInteraction(Interaction):
    """Catch-all interaction type for non-standard contacts."""
    pass


@dataclass
class Entry(Sequence):
    """Single BPSEQ entry (index, base, pairing partner)."""

    index_: int
    sequence: str
    pair: int

    def __getitem__(self, item):
        """Support tuple-like access to (index, sequence, pair)."""
        if item == 0:
            return self.index_
        elif item == 1:
            return self.sequence
        elif item == 2:
            return self.pair
        raise IndexError()

    def __lt__(self, other):
        """Order entries by index."""
        return self.index_ < other.index_

    def __len__(self) -> int:
        """Always return length 3 (index, sequence, pair)."""
        return 3

    def __str__(self):
        """Format entry in classic BPSEQ style: 'i base j'."""
        return f"{self.index_} {self.sequence} {self.pair}"


@dataclass(frozen=True)
class Strand:
    """Continuous strand segment with sequence and dot-bracket structure."""

    first: int
    last: int
    sequence: str
    structure: str

    @staticmethod
    def from_bpseq_entries(
        entries: List[Entry], dotbracket: str, reverse: bool = False
    ):
        """Build a Strand from a list of BPSEQ entries.

        Args:
            entries: Consecutive BPSEQ entries forming the strand.
            dotbracket: Full dot-bracket structure string.
            reverse: If True, reverse 5'–3' direction.

        Returns:
            New strand object covering the selected region.
        """

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
        """Return simple text representation of the strand."""
        return f"{self.first}-{self.sequence}-{self.last}"


@dataclass
class SingleStrand:
    """Single-stranded region (5', 3' or internal)."""

    strand: Strand
    is5p: bool
    is3p: bool

    def __post_init__(self):
        """Cache a human-readable description string."""
        self.description = str(self)

    def __str__(self):
        """Return a readable label describing the single strand."""
        if self.is5p:
            return f"SingleStrand5p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        if self.is3p:
            return f"SingleStrand3p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        return f"SingleStrand {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass
class Stem:
    """Paired stem defined by two complementary strands."""

    strand5p: Strand
    strand3p: Strand

    @staticmethod
    def from_bpseq_entries(
        strand5p_entries: List[Entry], all_entries: List, dotbracket: str
    ):
        """Build a Stem from 5' strand entries and full BPSEQ list.

        Args:
            strand5p_entries: Entries of the 5' side of the stem.
            all_entries: All BPSEQ entries for the molecule.
            dotbracket: Full dot-bracket structure string.

        Returns:
            New stem object with 5' and 3' strands.
        """

        paired = set([entry[2] for entry in strand5p_entries])
        strand3p_entries = list(filter(lambda entry: entry[0] in paired, all_entries))
        return Stem(
            Strand.from_bpseq_entries(strand5p_entries, dotbracket),
            Strand.from_bpseq_entries(strand3p_entries, dotbracket),
        )

    def __post_init__(self):
        """Cache a human-readable description string."""
        self.description = str(self)

    def __str__(self):
        """Return a readable representation of the stem."""
        return (
            f"Stem {self.strand5p.first} {self.strand5p.last} "
            f"{self.strand5p.sequence} {self.strand5p.structure} "
            f"{self.strand3p.first} {self.strand3p.last} "
            f"{self.strand3p.sequence} {self.strand3p.structure}"
        )


@dataclass
class Hairpin:
    """Hairpin loop represented by a single strand."""

    strand: Strand

    def __post_init__(self):
        """Cache a human-readable description string."""
        self.description = str(self)

    def __str__(self):
        """Return a readable representation of the hairpin."""
        return f"Hairpin {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass
class Loop:
    """Multi-strand loop consisting of several contiguous strands."""

    strands: List[Strand]

    def __post_init__(self):
        """Cache a human-readable description string."""
        self.description = str(self)

    def __str__(self):
        """Return a readable representation of the loop."""
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
    """Sequence and base-pairing information in BPSEQ format."""

    entries: List[Entry]

    @staticmethod
    def from_string(bpseq_str: str):
        """Parse BPSEQ-formatted text into a BpSeq object.

        Args:
            bpseq_str: Text containing BPSEQ lines.

        Returns:
            Parsed BPSEQ representation.
        """
        entries = []
        for line in bpseq_str.splitlines():
            line = line.strip()
            if len(line) == 0:
                continue
            fields = line.split()
            if len(fields) != 3:
                logging.warning("Failed to find 3 columns in BpSeq line: {}", line)
                continue
            entry = Entry(int(fields[0]), fields[1], int(fields[2]))
            entries.append(entry)
        return BpSeq(entries)

    @staticmethod
    def from_file(bpseq_path: str):
        """Read BPSEQ data from a file path.

        Args:
            bpseq_path: Path to a BPSEQ file.

        Returns:
            Parsed BPSEQ representation.
        """
        with open(bpseq_path) as f:
            return BpSeq.from_string(f.read())

    @staticmethod
    def from_dotbracket(dot_bracket):
        """Convert dot-bracket representation to BPSEQ entries.

        Args:
            dot_bracket: Dot-bracket representation with sequence and structure.

        Returns:
            BPSEQ representation derived from dot-bracket.
        """
        entries = [
            Entry(i + 1, dot_bracket.sequence[i], 0)
            for i in range(len(dot_bracket.sequence))
        ]
        for i, j in dot_bracket.pairs:
            entries[i].pair = j + 1
            entries[j].pair = i + 1
        return BpSeq(entries)

    def __post_init__(self):
        """Build internal mapping from indices to their pairing partners."""
        self.pairs = {}
        for i, _, j in self.entries:
            if j != 0:
                self.pairs[i] = j
                self.pairs[j] = i

    def __str__(self):
        """Format BPSEQ entries as multi-line text."""
        return "\n".join(("{} {} {}".format(i, c, j) for i, c, j in self.entries))

    def __eq__(self, other):
        """Compare two BpSeq objects entry by entry."""
        return len(self.entries) == len(other.entries) and all(
            ei == ej for ei, ej in zip(self.entries, other.entries)
        )

    @cached_property
    def sequence(self) -> str:
        """Return the nucleotide sequence as a string."""
        return "".join(entry.sequence for entry in self.entries)

    def paired(self, only5to3: bool = False):
        """Iterate over paired entries.

        Args:
            only5to3: If True, keep only pairs where index < partner.

        Returns:
            Iterator over paired Entry objects.
        """
        result = filter(lambda entry: entry.pair != 0, self.entries)
        if only5to3:
            result = filter(lambda entry: entry.index_ < entry.pair, result)
        return result

    @cached_property
    def __stems_entries(self) -> List[List[Entry]]:
        """Internal helper: group paired entries into stems."""
        stems = []
        entries: List[Entry] = []

        for entry in self.paired(only5to3=True):
            if not entries:
                entries.append(entry)
                continue

            i, _, j = entry
            k, _, l = entries[-1]
            if i == k + 1 and j == l - 1:
                entries.append(entry)
                continue

            stems.append(entries)
            entries = [entry]

        if entries:
            stems.append(entries)

        return stems

    @cached_property
    def elements(
        self,
    ) -> Tuple[List[Stem], List[SingleStrand], List[Hairpin], List[Loop]]:
        """Decompose structure into stems, single strands, hairpins and loops.

        Returns:
            Four lists describing each structural element type.
        """
        if not self.__stems_entries:
            return [], [], [], []

        stems, single_strands, hairpins, loops = [], [], [], []
        stopset = set()

        # stems
        for stem_entries in self.__stems_entries:
            stem = Stem.from_bpseq_entries(
                stem_entries, self.entries, self.dot_bracket.structure
            )
            stems.append(stem)
            stopset.add(stem.strand5p.first - 1)
            stopset.add(stem.strand5p.last - 1)
            stopset.add(stem.strand3p.first - 1)
            stopset.add(stem.strand3p.last - 1)

        stops = sorted(stopset)
        loop_candidates = []

        # 5' single strand
        if stops[0] > 0:
            single_strands.append(
                SingleStrand(
                    Strand.from_bpseq_entries(
                        self.entries[: stops[0] + 1],
                        self.dot_bracket.structure,
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
                    hairpins.append(
                        Hairpin(
                            Strand.from_bpseq_entries(
                                candidate, self.dot_bracket.structure
                            )
                        )
                    )
                else:
                    loop_candidates.append(
                        Strand.from_bpseq_entries(candidate, self.dot_bracket.structure)
                    )

        # 3' single strand
        if stops[-1] < len(self.entries) - 1:
            single_strands.append(
                SingleStrand(
                    Strand.from_bpseq_entries(
                        self.entries[stops[-1] :], self.dot_bracket.structure
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

            while True:
                for j in graph[i]:
                    if (
                        loop_candidates[j] not in used
                        and loop_candidates[j] not in loop
                    ):
                        loop.append(loop_candidates[j])
                        i = j
                        break
                else:
                    break

            if self.entries[loop[0].first - 1].pair == loop[-1].last:
                if not all([strand.last - strand.first <= 1 for strand in loop]):
                    loops.append(Loop(loop))
                    used.update(loop)

        for loop_candidate in loop_candidates:
            if loop_candidate not in used:
                single_strands.append(SingleStrand(loop_candidate, False, False))

        return stems, single_strands, hairpins, loops

    @cached_property
    def graphviz(self):
        """Create and render a Graphviz graph of structural elements."""
        stems, single_strands, hairpins, loops = self.elements
        graph = defaultdict(set)
        dot = graphviz.Graph()

        for single_strand in single_strands:
            graph[str(single_strand)].update(
                [
                    single_strand.strand.first,
                    single_strand.strand.last,
                ]
            )

        for stem in stems:
            if stem.strand5p.first == stem.strand5p.last:
                continue
            graph[str(stem)].update(
                [
                    stem.strand5p.first,
                    stem.strand5p.last,
                    stem.strand3p.first,
                    stem.strand3p.last,
                ]
            )

        for hairpin in hairpins:
            graph[str(hairpin)].update(
                [
                    hairpin.strand.first,
                    hairpin.strand.last,
                ]
            )

        for loop in loops:
            stops = set()
            for strand in loop.strands:
                stops.update(
                    [
                        strand.first,
                        strand.last,
                    ]
                )
            graph[str(loop)].update(stops)

        for i, element in enumerate(graph.keys()):
            dot.node(f"E{i}", str(element))

        keys = list(graph.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if graph[keys[i]].intersection(graph[keys[j]]):
                    dot.edge(f"E{i}", f"E{j}")

        return dot.render()

    @cached_property
    def __regions(self) -> List[Tuple[int, int, int]]:
        """Internal helper: list of stem regions (start, partner, length)."""
        return [
            (stem_entries[0].index_, stem_entries[0].pair, len(stem_entries))
            for stem_entries in self.__stems_entries
        ]

    @cached_property
    def dot_bracket(self):
        """Convert BPSEQ information to dot-bracket using MILP solver if available."""
        if pulp.HiGHS_CMD().available():
            solver = pulp.HiGHS_CMD()  # much faster than default
        else:
            solver = pulp.LpSolverDefault
        if solver is not None:
            solver.msg = False
        return self.convert_to_dot_bracket(solver)

    def convert_to_dot_bracket(self, solver: pulp.LpSolver):
        """Convert BPSEQ to dot-bracket using a given PuLP solver.

        If the solver is not available or the problem is infeasible,
        the method falls back to a greedy FCFS algorithm.

        Args:
            solver: PuLP MILP solver instance or None.

        Returns:
            DotBracket: Dot-bracket representation of the structure.
        """
        # if PuLP solvers are not installed, use FCFS
        if solver is None:
            return self.fcfs()

        # build conflict graph
        regions = self.__regions
        graph = defaultdict(set)

        for i, j in itertools.combinations(range(len(regions)), 2):
            ri, rj = regions[i], regions[j]
            k, l, _ = ri
            m, n, _ = rj

            # is pseudoknot?
            if (k < m < l < n) or (m < k < n < l):
                graph[i].add(j)
                graph[j].add(i)

        # return all non-pseudoknotted if the graph is empty
        if not graph:
            return self.__make_dot_bracket(regions, [0 for _ in range(len(regions))])

        # determine maximum pseudoknot order as chromatic number bound equal to maximum vertex degree + 1
        max_order = max(map(len, graph.values())) + 1

        # define the problem
        problem = pulp.LpProblem("POA", pulp.LpMaximize)

        # create decision variables
        variables = []
        vars_by_region = defaultdict(list)
        vars_by_order = defaultdict(list)
        var_by_region_order = {}
        region_by_var = {}
        for i in range(len(regions)):
            for j in range(max_order):
                variable = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpInteger)
                variables.append(variable)
                vars_by_region[i].append(variable)
                vars_by_order[j].append(variable)
                var_by_region_order[(i, j)] = variable
                region_by_var[variable] = regions[i]

        # define objective function terms
        terms = []

        for order, vars in vars_by_order.items():
            for var in vars:
                length = region_by_var[var][2]
                if order == 0:
                    terms.append(var * length)
                else:
                    terms.append(-1 * var * length * order)

        # define objective function
        problem += pulp.lpSum(terms)

        # define constraints that each region is assigned to exactly one order
        for region_vars in vars_by_region.values():
            problem += pulp.lpSum(region_vars) == 1

        # define constraints that no two adjacent regions are assigned to the same order
        for i in graph.keys():
            for j in graph[i]:
                for order in range(max_order):
                    problem += (
                        var_by_region_order[(i, order)]
                        + var_by_region_order[(j, order)]
                        <= 1
                    )

        # solve the problem
        try:
            logging.debug(f"POA: problem formulation\n{problem}")
            problem.solve(solver)
        except pulp.PulpSolverError:
            logging.warning(
                "POA: failed to solve problem using MILP approach, fallback to FCFS"
            )
            return self.fcfs()

        # if problem is infeasible, fallback to FCFS
        if problem.status != pulp.LpStatusOptimal:
            logging.warning("POA: problem is infeasible, fallback to FCFS")
            return self.fcfs()

        # log solver time statistics
        logging.debug(
            f"POA: solver {solver.name} took {round(problem.solutionTime, 2)} seconds"
        )

        # map variable values to orders
        orders = [0 for _ in range(len(regions))]
        for variable in problem.variables():
            if variable.varValue == 1:
                name = variable.getName()
                i, order = map(int, name.split("_")[1:])
                orders[i] = order

        return self.__make_dot_bracket(regions, orders)

    def __make_dot_bracket(self, regions, orders):
        """Internal helper: build final dot-bracket object from regions and orders.

        Returns:
            Dot-bracket representation built from the given stem regions and orders.
        """
        # build dot-bracket
        sequence = self.sequence
        structure = ["." for _ in range(len(sequence))]
        brackets = ["()", "[]", "{}", "<>"] + [
            "".join(p) for p in zip(string.ascii_uppercase, string.ascii_lowercase)
        ]

        for i, stem in enumerate(regions):
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

    @cached_property
    def fcfs(self):
        """Greedy FCFS (first-come, first-served) conversion to dot-bracket.

        Returns:
            Dot-bracket representation produced by the FCFS heuristic.
        """
        regions = [
            (stem_entries[0].index_, stem_entries[0].pair, len(stem_entries))
            for stem_entries in self.__stems_entries
        ]
        orders = [0 for i in range(len(regions))]

        for i in range(1, len(regions)):
            k, l, _ = regions[i]
            available = [True for _ in range(len("([{<" + string.ascii_uppercase))]

            for j in range(i):
                m, n, _ = regions[j]
                conflicted = (k < m < l < n) or (m < k < n < l)

                if conflicted:
                    available[orders[j]] = False

            order = next(filter(lambda i: available[i] is True, range(len(available))))
            orders[i] = order

        return self.__make_dot_bracket(regions, orders)

    @cached_property
    def all_dot_brackets(self):
        """Enumerate all valid dot-bracket solutions for pseudoknotted structures.

        Returns:
            List of possible dot-bracket assignments.
        """
        # build conflict graph
        regions = self.__regions
        graph = defaultdict(set)

        for i, j in itertools.combinations(range(len(regions)), 2):
            ri, rj = regions[i], regions[j]
            k, l, _ = ri
            m, n, _ = rj

            # is pseudoknot?
            if (k < m < l < n) or (m < k < n < l):
                graph[i].add(j)
                graph[j].add(i)

        # early exit for non-pseudoknotted structures
        vertices = list(graph.keys())
        if not vertices:
            return [self.fcfs]

        # find all connected components
        visited = {vertex: False for vertex in vertices}
        components = []

        for vertex in vertices:
            if not visited[vertex]:
                visited[vertex] = True
                stack = [vertex]
                components.append([vertex])

                while stack:
                    current = stack[-1]
                    next_vertex = None

                    for neighbor in graph[current]:
                        if not visited[neighbor]:
                            next_vertex = neighbor
                            break

                    if next_vertex is not None:
                        visited[next_vertex] = True
                        stack.append(next_vertex)
                        components[-1].append(next_vertex)
                    else:
                        stack.pop()

        # find unique orders for each component
        unique = []
        for component in components:
            unique.append(set())

            for permutation in itertools.permutations(component):
                orders = {region: 0 for region in component}

                for i in range(1, len(permutation)):
                    available = [True for _ in range(len(component))]

                    for j in range(i):
                        if permutation[j] in graph[permutation[i]]:
                            available[orders[permutation[j]]] = False

                    order = next(
                        filter(lambda k: available[k] is True, range(len(available)))
                    )
                    orders[permutation[i]] = order

                unique[-1].add(frozenset(orders.items()))

        # generate all possible dot-brackets
        solutions = set()
        for assignment in itertools.product(*unique):
            orders = {region: 0 for region in range(len(regions))}

            for order in assignment:
                orders.update(order)

            solutions.add(self.__make_dot_bracket(regions, orders))
        return list(solutions)

    def without_pseudoknots(self):
        """Return BPSEQ converted to dot-bracket with pseudoknots removed.

        Returns:
            New BPSEQ object with all pseudoknots removed.
        """
        return BpSeq.from_dotbracket(self.dot_bracket.without_pseudoknots())

    def without_isolated(self):
        """Return BpSeq with isolated base pairs unpaired.

        Returns:
            New BPSEQ object where isolated base pairs have been unpaired.
        """
        stems, _, _, _ = self.elements
        to_unpair = []

        for stem in stems:
            if stem.strand5p.first == stem.strand5p.last:
                to_unpair.append(stem.strand5p.first - 1)
                to_unpair.append(stem.strand3p.first - 1)

        if not to_unpair:
            return self

        entries = self.entries.copy()
        for i in to_unpair:
            entries[i].pair = 0

        return BpSeq(entries)


@dataclass
class DotBracket:
    """Sequence and structure in dot-bracket notation."""

    sequence: str
    structure: str

    @staticmethod
    def from_file(path: str):
        """Read DotBracket from a file with 2–3 lines.

        Args:
            path: Path to a file with sequence and structure lines.

        Returns:
            Parsed dot-bracket object.
        """
        with open(path) as f:
            lines = f.readlines()
        if len(lines) == 2:
            return DotBracket.from_string(lines[0].rstrip(), lines[1].rstrip())
        if len(lines) == 3:
            return DotBracket.from_string(lines[1].rstrip(), lines[2].rstrip())
        raise RuntimeError(f"Failed to read DotBracket from file: {path}")

    @staticmethod
    def from_string(sequence: str, structure: str):
        """Create a DotBracket object from raw sequence and structure strings.

        Args:
            sequence: Nucleotide sequence.
            structure: Dot-bracket string of the same length.

        Returns:
            New dot-bracket object.

        Raises:
            ValueError: If sequence and structure lengths differ.
        """
        if len(sequence) != len(structure):
            raise ValueError(
                "Sequence and structure lengths differ, {} vs {}",
                (len(sequence), len(structure)),
            )
        return DotBracket(sequence, structure)

    def __post_init__(self):
        """Parse structure string into a list of paired indices."""
        self.pairs = []

        opening = "([{<" + string.ascii_uppercase
        closing = ")]}>" + string.ascii_lowercase
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
        """Return sequence and structure as two lines of text."""
        return f"{self.sequence}\n{self.structure}"

    def __eq__(self, other):
        """Compare dot-bracket objects by sequence and structure."""
        return self.sequence == other.sequence and self.structure == other.structure

    def __hash__(self) -> int:
        """Hash dot-bracket based on sequence and structure."""
        return hash((self.sequence, self.structure))

    def without_pseudoknots(self):
        """Return a copy with pseudoknot brackets replaced by dots.

        Returns:
            New dot-bracket object with pseudoknots removed.
        """
        structure = re.sub(r"[\[\]\{\}\<\>A-Za-z]", ".", self.structure)
        return DotBracket(self.sequence, structure)


@dataclass
class MultiStrandDotBracket(DotBracket):
    """Dot-bracket representation for multiple strands concatenated together."""

    strands: List[Strand]

    @staticmethod
    def from_string(sequence: str, structure: str):
        """Create MultiStrandDotBracket from a single sequence/structure.

        For compatibility with DotBracket, this creates a single strand
        covering the full sequence.

        Args:
            sequence: Nucleotide sequence.
            structure: Dot-bracket structure string.

        Returns:
            Multi-strand dot-bracket object.
        """
        # Provide compatibility with DotBracket.from_string
        strand = Strand(1, len(sequence), sequence, structure)
        return MultiStrandDotBracket(sequence, structure, [strand])

    @staticmethod
    def from_multiline_string(input: str):
        """Parse multi-strand dot-bracket from a multi-line string.

        The format expects optional header lines starting with '>', followed
        by sequence and structure lines for each strand.

        Args:
            input: Full multi-line text.

        Returns:
            Parsed multi-strand representation.
        """
        strands = []
        first = 1

        for match in re.finditer(
            r"((>.*?\n)?([ACGTURYSWKMBDHVNacgturyswkmbdhvn.-]+)\n([.()\[\]{}<>A-Za-z]+))",
            input,
        ):
            sequence = match.group(3)
            structure = match.group(4)
            assert len(sequence) == len(structure)
            last = first + len(sequence) - 1
            strands.append(Strand(first, last, sequence, structure))
            first = last + 1

        return MultiStrandDotBracket(
            "".join(strand.sequence for strand in strands),
            "".join(strand.structure for strand in strands),
            strands,
        )

    @staticmethod
    def from_file(path: str):
        """Read MultiStrandDotBracket from a file.

        Args:
            path: Path to the input file.

        Returns:
            Parsed multi-strand dot-bracket object.
        """
        with open(path) as f:
            return MultiStrandDotBracket.from_multiline_string(f.read())


@dataclass(frozen=True, order=True)
class BaseInteractions:
    """Container for all base-level interactions in a structure."""

    base_pairs: List[BasePair]
    stackings: List[Stacking]
    base_ribose_interactions: List[BaseRibose]
    base_phosphate_interactions: List[BasePhosphate]
    other_interactions: List[OtherInteraction]

    @classmethod
    def from_structure3d(
        cls,
        structure3d: "Structure3D",
        base_pairs: List[BasePair],
        stackings: List[Stacking],
        base_ribose_interactions: List[BaseRibose],
        base_phosphate_interactions: List[BasePhosphate],
        other_interactions: List[OtherInteraction],
    ) -> "BaseInteractions":
        """Unify residue identifiers across interactions based on a 3D structure.

        This method ensures that all interactions share consistent auth/label
        coordinates and, when possible, fills in missing Saenger classes.

        Args:
            structure3d: 3D structure used as a reference.
            base_pairs: List of base-pair interactions.
            stackings: List of stacking interactions.
            base_ribose_interactions: List of base–ribose interactions.
            base_phosphate_interactions: List of base–phosphate interactions.
            other_interactions: List of other interactions.

        Returns:
            BaseInteractions: Normalized interactions container.
        """
        cni2residue = {}
        cni2label = {}
        cni2auth = {}

        for residue3d in structure3d.residues:
            cni = (residue3d.chain, residue3d.number, residue3d.icode or None)
            cni2auth[cni] = residue3d.auth
            cni2label[cni] = residue3d.label
            cni2residue[cni] = residue3d

        def unify_nt(nt: Residue) -> Residue:
            if nt.auth is not None and nt.label is not None:
                return nt
            cni = (nt.chain, nt.number, nt.icode or None)
            if nt.auth is not None:
                return Residue(label=cni2label.get(cni, None), auth=nt.auth)
            if nt.label is not None:
                return Residue(label=nt.label, auth=cni2auth.get(cni, None))
            return nt

        base_pairs_new = []
        for base_pair in base_pairs:
            nt1 = unify_nt(base_pair.nt1)
            nt2 = unify_nt(base_pair.nt2)

            cni1 = (nt1.chain, nt1.number, nt1.icode or None)
            cni2 = (nt2.chain, nt2.number, nt2.icode or None)
            if cni1 not in cni2residue or cni2 not in cni2residue:
                saenger = base_pair.saenger
            else:
                saenger = base_pair.saenger or Saenger.from_leontis_westhof(
                    cni2residue[cni1].one_letter_name,
                    cni2residue[cni2].one_letter_name,
                    base_pair.lw,
                )
            if (
                nt1 != base_pair.nt1
                or nt2 != base_pair.nt2
                or saenger != base_pair.saenger
            ):
                base_pair = BasePair(nt1=nt1, nt2=nt2, lw=base_pair.lw, saenger=saenger)
            base_pairs_new.append(base_pair)

        stackings_new = []
        for stacking in stackings:
            nt1 = unify_nt(stacking.nt1)
            nt2 = unify_nt(stacking.nt2)
            if nt1 != stacking.nt1 or nt2 != stacking.nt2:
                stacking = Stacking(nt1=nt1, nt2=nt2, topology=stacking.topology)
            stackings_new.append(stacking)

        base_ribose_interactions_new = []
        for base_ribose in base_ribose_interactions:
            nt1 = unify_nt(base_ribose.nt1)
            nt2 = unify_nt(base_ribose.nt2)
            if nt1 != base_ribose.nt1 or nt2 != base_ribose.nt2:
                base_ribose = BaseRibose(nt1=nt1, nt2=nt2, br=base_ribose.br)
            base_ribose_interactions_new.append(base_ribose)

        base_phosphate_interactions_new = []
        for base_phosphate in base_phosphate_interactions:
            nt1 = unify_nt(base_phosphate.nt1)
            nt2 = unify_nt(base_phosphate.nt2)
            if nt1 != base_phosphate.nt1 or nt2 != base_phosphate.nt2:
                base_phosphate = BasePhosphate(nt1=nt1, nt2=nt2, bph=base_phosphate.bph)
            base_phosphate_interactions_new.append(base_phosphate)

        other_interactions_new = []
        for other_interaction in other_interactions:
            nt1 = unify_nt(other_interaction.nt1)
            nt2 = unify_nt(other_interaction.nt2)
            if nt1 != other_interaction.nt1 or nt2 != other_interaction.nt2:
                other_interaction = OtherInteraction(nt1=nt1, nt2=nt2)
            other_interactions_new.append(other_interaction)

        return cls(
            base_pairs=base_pairs_new,
            stackings=stackings_new,
            base_ribose_interactions=base_ribose_interactions_new,
            base_phosphate_interactions=base_phosphate_interactions_new,
            other_interactions=other_interactions_new,
        )


@dataclass(frozen=True, order=True)
class InterStemParameters:
    """Geometric parameters describing relationships between two stems."""

    stem1_idx: int
    stem2_idx: int
    type: Optional[str]  # Type of closest endpoint pair ('cs55', 'cs53', etc.)
    torsion: Optional[float]  # Torsion angle between stem segments (degrees)
    min_endpoint_distance: Optional[float]  # Minimum distance between stem endpoints
    torsion_angle_pdf: Optional[float]  # PDF value of the torsion angle
    min_endpoint_distance_pdf: Optional[float]  # PDF value of the min endpoint distance
    coaxial_probability: Optional[float]  # Probability of stems being coaxial (0-1)


@dataclass
class Structure2D:
    """Secondary structure representation plus derived structural elements.

    This object collects interactions, BPSEQ, dot-bracket notation,
    stems, loops and optional inter-stem parameters.
    """

    base_pairs: List[BasePair]
    stackings: List[Stacking]
    base_ribose_interactions: List[BaseRibose]
    base_phosphate_interactions: List[BasePhosphate]
    other_interactions: List[OtherInteraction]
    bpseq: BpSeq
    bpseq_index: Dict[int, Residue]
    dot_bracket: MultiStrandDotBracket
    extended_dot_bracket: str
    stems: List[Stem]
    single_strands: List[SingleStrand]
    hairpins: List[Hairpin]
    loops: List[Loop]
    inter_stem_parameters: List[InterStemParameters]
