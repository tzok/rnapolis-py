from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import math

import numpy
import numpy.typing
from rnapolis.common import LeontisWesthof
from rnapolis.secondary import Structure2D

from rnapolis.tertiary import Atom, BasePair3D, Structure3D, Residue3D, Stacking3D


@dataclass
class Mapping2D3D:
    structure3d: Structure3D
    structure2d: Structure2D
    base_pairs: List[BasePair3D] = field(init=False)
    base_pair_graph: Dict[Residue3D, Set[Tuple[Residue3D, LeontisWesthof]]] = field(
        init=False
    )
    stackings: List[Stacking3D] = field(init=False)
    stacking_graph: Dict[Residue3D, Set[Residue3D]] = field(init=False)

    def __post_init__(self):
        self.base_pairs = self.__base_pairs()
        self.base_pair_graph = self.__base_pair_graph()
        self.stackings = self.__stackings()
        self.stacking_graph = self.__stacking_graph()

    def __base_pairs(self) -> List[BasePair3D]:
        result = []
        for base_pair in self.structure2d.basePairs:
            nt1 = self.structure3d.find_residue(base_pair.nt1.label, base_pair.nt1.auth)
            nt2 = self.structure3d.find_residue(base_pair.nt2.label, base_pair.nt2.auth)
            if nt1 is not None and nt2 is not None:
                result.append(
                    BasePair3D(
                        base_pair.nt1,
                        base_pair.nt2,
                        base_pair.lw,
                        base_pair.saenger,
                        nt1,
                        nt2,
                    )
                )
        return result

    def __base_pair_graph(
        self,
    ) -> Dict[Residue3D, Set[Tuple[Residue3D, LeontisWesthof]]]:
        graph = defaultdict(set)
        for pair in self.base_pairs:
            graph[pair.nt1].add((pair.nt2, pair.lw))
            graph[pair.nt2].add((pair.nt1, pair.lw.reverse))
        return graph

    def __stackings(self) -> List[Stacking3D]:
        result = []
        for stacking in self.structure2d.stackings:
            nt1 = self.structure3d.find_residue(stacking.nt1.label, stacking.nt1.auth)
            nt2 = self.structure3d.find_residue(stacking.nt2.label, stacking.nt2.auth)
            if nt1 is not None and nt2 is not None:
                result.append(
                    Stacking3D(stacking.nt1, stacking.nt2, stacking.topology, nt1, nt2)
                )
        return result

    def __stacking_graph(self) -> Dict[Residue3D, Set[Residue3D]]:
        graph = defaultdict(set)
        for pair in self.stackings:
            graph[pair.nt1].add(pair.nt2)
            graph[pair.nt2].add(pair.nt1)
        return graph


def torsion_angle(a1: Atom, a2: Atom, a3: Atom, a4: Atom) -> float:
    v1 = a2.coordinates - a1.coordinates
    v2 = a3.coordinates - a2.coordinates
    v3 = a4.coordinates - a3.coordinates
    t1: numpy.typing.NDArray[numpy.floating] = numpy.cross(v1, v2)
    t2: numpy.typing.NDArray[numpy.floating] = numpy.cross(v2, v3)
    t3: numpy.typing.NDArray[numpy.floating] = v1 * numpy.linalg.norm(v2)
    return math.atan2(numpy.dot(t2, t3), numpy.dot(t1, t2))


def angle_between_vectors(
    v1: numpy.typing.NDArray[numpy.floating], v2: numpy.typing.NDArray[numpy.floating]
) -> float:
    return math.acos(numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2))
