from dataclasses import dataclass
from typing import List, Optional

from rnapolis.common import BR, BPh, LeontisWesthof, Residue, Saenger, StackingTopology


@dataclass
class Interaction:
    nt1: Residue
    nt2: Residue

    def __post_init__(self):
        if isinstance(self.nt1, dict):
            self.nt1 = Residue(**self.nt1)
        if isinstance(self.nt2, dict):
            self.nt2 = Residue(**self.nt2)


@dataclass
class BasePair(Interaction):
    lw: LeontisWesthof
    saenger: Optional[Saenger]

    def __post_init__(self):
        super(BasePair, self).__post_init__()
        if isinstance(self.lw, str):
            self.lw = LeontisWesthof[self.lw]
        if isinstance(self.saenger, str):
            self.saenger = Saenger[self.saenger]


@dataclass
class Stacking(Interaction):
    topology: Optional[StackingTopology]

    def __post_init__(self):
        super(Stacking, self).__post_init__()
        if isinstance(self.topology, str):
            self.topology = StackingTopology[self.topology]


@dataclass
class BaseRibose(Interaction):
    br: Optional[BR]

    def __post_init__(self):
        super(BaseRibose, self).__post_init__()
        if isinstance(self.br, str):
            self.br = BR[self.br]


@dataclass
class BasePhosphate(Interaction):
    bph: Optional[BPh]

    def __post_init__(self):
        super(BasePhosphate, self).__post_init__()
        if isinstance(self.bph, str):
            self.bph = BPh[self.bph]


@dataclass
class OtherInteraction(Interaction):
    def __post_init__(self):
        super(OtherInteraction, self).__post_init__()


@dataclass
class Structure2D:
    basePairs: List[BasePair]
    stackings: List[Stacking]
    baseRiboseInteractions: List[BaseRibose]
    basePhosphateInteractions: List[BasePhosphate]
    otherInteractions: List[OtherInteraction]

    def __post_init__(self):
        self.basePairs = [
            BasePair(**x) if isinstance(x, dict) else x for x in self.basePairs
        ]
        self.stackings = [
            Stacking(**x) if isinstance(x, dict) else x for x in self.stackings
        ]
        self.baseRiboseInteractions = [
            BaseRibose(**x) if isinstance(x, dict) else x
            for x in self.baseRiboseInteractions
        ]
        self.basePhosphateInteractions = [
            BasePhosphate(**x) if isinstance(x, dict) else x
            for x in self.basePhosphateInteractions
        ]
        self.otherInteractions = [
            OtherInteraction(**x) if isinstance(x, dict) else x
            for x in self.otherInteractions
        ]
