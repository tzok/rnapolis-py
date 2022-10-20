import math

from rnapolis.tertiary import Atom, torsion_angle


def test_torsion_angle():
    a1 = Atom(None, None, 1, "P", 50.63, 49.73, 50.57, None)
    a2 = Atom(None, None, 1, "O5'", 50.16, 49.14, 52.02, None)
    a3 = Atom(None, None, 1, "C5'", 50.22, 49.95, 53.21, None)
    a4 = Atom(None, None, 1, "C4'", 50.97, 49.23, 54.31, None)
    assert math.isclose(
        math.degrees(torsion_angle(a1, a2, a3, a4)), -127.83976634524326
    )
