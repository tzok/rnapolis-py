import itertools
import logging
import os
import string
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property, total_ordering
from typing import Dict, List, Optional, Tuple

import graphviz
import pulp

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)


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


@dataclass
class SingleStrand:
    strand: Strand
    is5p: bool
    is3p: bool

    def __post_init__(self):
        self.description = str(self)

    def __str__(self):
        if self.is5p:
            return f"SingleStrand5p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        if self.is3p:
            return f"SingleStrand3p {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"
        return f"SingleStrand {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass
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

    def __post_init__(self):
        self.description = str(self)

    def __str__(self):
        return f"Stem {self.strand5p.first} {self.strand5p.last} {self.strand5p.sequence} {self.strand5p.structure} {self.strand3p.first} {self.strand3p.last} {self.strand3p.sequence} {self.strand3p.structure}"


@dataclass
class Hairpin:
    strand: Strand

    def __post_init__(self):
        self.description = str(self)

    def __str__(self):
        return f"Hairpin {self.strand.first} {self.strand.last} {self.strand.sequence} {self.strand.structure}"


@dataclass
class Loop:
    strands: List[Strand]

    def __post_init__(self):
        self.description = str(self)

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
    def from_string(bpseq_str: str):
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
        with open(bpseq_path) as f:
            return BpSeq.from_string(f.read())

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
    def __stems_entries(self) -> List[List[Entry]]:
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
        return [
            (stem_entries[0].index_, stem_entries[0].pair, len(stem_entries))
            for stem_entries in self.__stems_entries
        ]

    @cached_property
    def dot_bracket(self):
        pulp.LpSolverDefault.msg = False
        return self.convert_to_dot_bracket(pulp.LpSolverDefault)

    def convert_to_dot_bracket(self, solver: pulp.LpSolver):
        # if PuLP solvers are not installed, use FCFS
        if len(pulp.listSolvers(onlyAvailable=True)) == 0:
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
        regions = [
            (stem_entries[0].index_, stem_entries[0].pair, len(stem_entries))
            for stem_entries in self.__stems_entries
        ]
        orders = [0 for i in range(len(regions))]

        for i in range(1, len(regions)):
            k, l, _ = regions[i]
            available = [True for i in range(10)]

            for j in range(i):
                m, n, _ = regions[j]
                conflicted = (k < m < l < n) or (m < k < n < l)

                if conflicted:
                    available[orders[j]] = False

            order = next(filter(lambda i: available[i] is True, range(len(available))))
            orders[i] = order

        return self.__make_dot_bracket(regions, orders)


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
        return f"{self.sequence}\n{self.structure}"


@dataclass(frozen=True, order=True)
class BaseInteractions:
    basePairs: List[BasePair]
    stackings: List[Stacking]
    baseRiboseInteractions: List[BaseRibose]
    basePhosphateInteractions: List[BasePhosphate]
    otherInteractions: List[OtherInteraction]


@dataclass(frozen=True, order=True)
class Structure2D:
    baseInteractions: BaseInteractions
    bpseq: str
    dotBracket: str
    extendedDotBracket: str
    stems: List[Stem]
    singleStrands: List[SingleStrand]
    hairpins: List[Hairpin]
    loops: List[Loop]
