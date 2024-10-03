from collections import Counter

from rnapolis.annotator import extract_base_interactions, extract_secondary_structure
from rnapolis.parser import read_3d_structure


def test_1ehz():
    """
    Make sure there are no duplicates on any list
    """
    with open("tests/1ehz-assembly-1.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d, 1)

    interactions = [
        base_interactions.basePairs,
        base_interactions.stackings,
        base_interactions.baseRiboseInteractions,
        base_interactions.basePhosphateInteractions,
        base_interactions.otherInteractions,
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
                assert False, f"Interaction {element} occurs {count} times among {labels[i]} type: {duplicates}"


def test_8btk():
    with open("tests/8btk_B7.cif") as f:
        structure3d = read_3d_structure(f, 1)
    assert extract_secondary_structure(structure3d, 1) is not None


def test_488d():
    """
    There are clashing residues 151 in chains B and D. The clash is caused by occupancy factors less than 1.
    """
    with open("tests/488d.pdb") as f:
        structure3d = read_3d_structure(f)

    base_interactions = extract_base_interactions(structure3d)
    assert base_interactions is not None
