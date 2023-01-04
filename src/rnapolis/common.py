import string
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, total_ordering
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

    @cached_property
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

    @cached_property
    def is_canonical(self) -> bool:
        return self == Saenger.XIX or self == Saenger.XX or self == Saenger.XXVIII


class StackingTopology(Enum):
    upward = "upward"
    downward = "downward"
    inward = "inward"
    outward = "outward"

    @cached_property
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

    @cached_property
    def chain(self) -> str:
        if self.auth is not None:
            return self.auth.chain
        if self.label is not None:
            return self.label.chain
        raise RuntimeError(
            "Unknown chain name, both ResidueAuth and ResidueLabel are empty"
        )

    @cached_property
    def number(self) -> int:
        if self.auth is not None:
            return self.auth.number
        if self.label is not None:
            return self.label.number
        raise RuntimeError(
            "Unknown residue number, both ResidueAuth and ResidueLabel are empty"
        )

    @cached_property
    def icode(self) -> Optional[str]:
        if self.auth is not None:
            return self.auth.icode if self.auth.icode not in (" ", "?") else None
        return None

    @cached_property
    def name(self) -> str:
        if self.auth is not None:
            return self.auth.name
        if self.label is not None:
            return self.label.name
        raise RuntimeError(
            "Unknown residue name, both ResidueAuth and ResidueLabel are empty"
        )

    @cached_property
    def molecule_type(self) -> Molecule:
        if self.name.upper() in ("A", "C", "G", "U"):
            return Molecule.RNA
        if self.name.upper() in ("DA", "DC", "DG", "DT"):
            return Molecule.DNA
        return Molecule.Other

    @cached_property
    def full_name(self) -> str:
        if self.auth is not None:
            if self.auth.chain.isspace():
                builder = f"{self.auth.name}"
            else:
                builder = f"{self.auth.chain}.{self.auth.name}"
            if self.auth.name[-1] in string.digits:
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
            if self.label.name[-1] in string.digits:
                builder += "/"
            builder += f"{self.label.number}"
            return builder
        raise RuntimeError(
            "Unknown full residue name, both ResidueAuth and ResidueLabel are empty"
        )


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
