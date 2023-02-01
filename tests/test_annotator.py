from collections import Counter

from rnapolis.annotator import extract_secondary_structure
from rnapolis.common import BR
from rnapolis.parser import read_3d_structure


def test_1ehz():
    """
    Make sure there are no duplicates on any list
    """
    with open("tests/1ehz-assembly-1.cif") as f:
        structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d, 1)

    interactions = [
        structure2d.basePairs,
        structure2d.stackings,
        structure2d.baseRiboseInteractions,
        structure2d.basePhosphateInteractions,
        structure2d.otherInteractions,
    ]
    labels = ["base pairs", "stackings", "base-ribose", "base-phosphate", "other"]
    for i in range(len(interactions)):
        counter = Counter(interactions[i])
        for element, count in counter.most_common():
            assert count == 1, f"Interaction {element} occurs {count} times"

        simple_collection = [
            (bp.nt1.full_name, bp.nt2.full_name) for bp in interactions[i]
        ]
        counter = Counter(simple_collection)
        for element, count in counter.most_common():
            if count != 1:
                duplicates = [
                    bp
                    for bp in interactions[i]
                    if (bp.nt1.full_name, bp.nt2.full_name) == element
                ]
                assert (
                    False
                ), f"Interaction {element} occurs {count} times among {labels[i]} type: {duplicates}"
