import math

import numpy
import numpy.typing

from rnapolis.tertiary import Atom


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
