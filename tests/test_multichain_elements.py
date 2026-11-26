"""Tests for chain-boundary-aware structural element decomposition.

Multi-chain structures are represented as a BpSeq that concatenates all
chains into one continuous index space.  Without strand-break information,
``compute_elements`` finds spurious hairpins and single strands that span
chain junctions.  These tests lock in the fixed behaviour: elements must
never cross a strand break, and inputs without ``strand_breaks`` keep the
legacy concatenation semantics.
"""

import copy

import orjson

from rnapolis.annotator import extract_base_interactions
from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BpSeq,
    DotBracket,
    Entry,
    LeontisWesthof,
    MultiStrandDotBracket,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Structure2D,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import (
    Mapping2D3D,
    Residue3D,
    Structure3D,
    extract_secondary_structure,
)


def build_bpseq(pairs, sequence, strand_breaks=None):
    """Build a BpSeq from pairing indices (1-based) and a sequence string."""
    pair_map = {}
    for i, j in pairs:
        pair_map[i] = j
        pair_map[j] = i
    entries = [
        Entry(index, sequence[index - 1], pair_map.get(index, 0))
        for index in range(1, len(sequence) + 1)
    ]
    return BpSeq(entries, strand_breaks=strand_breaks)


def single_strand_flags(single_strands):
    """Summarize single strands as (is5p, is3p, first, last) tuples."""
    return {
        (ss.is5p, ss.is3p, ss.strand.first, ss.strand.last)
        for ss in single_strands
    }


def assert_strands_do_not_cross_break(stems, single_strands, hairpins, loops, breaks):
    """No element strand may span any position in ``breaks`` (1-based)."""
    strands = []
    for stem in stems:
        strands += [stem.strand5p, stem.strand3p]
    strands += [ss.strand for ss in single_strands]
    strands += [hairpin.strand for hairpin in hairpins]
    for loop in loops:
        strands += loop.strands
    for strand in strands:
        for b in breaks:
            assert not (strand.first <= b < strand.last), (
                f"Strand {strand} crosses strand break at {b}"
            )


# ---------------------------------------------------------------------------
# Scenarios: chain A is concatenated with chain B into one index space,
# exactly like Mapping2D3D.__generate_bpseq does for multi-chain input.
# ---------------------------------------------------------------------------

# T1: blunt-end duplex, chain A = 1-8, chain B = 9-16
BLUNT_SEQUENCE = "G" * 8 + "C" * 8
BLUNT_PAIRS = [(i, 17 - i) for i in range(1, 9)]
BLUNT_BREAKS = [8]

# T2: duplex with unpaired 5' tails, chain A = 1-9, chain B = 10-18
STICKY_SEQUENCE = "AAAGGGGGG" + "AAACCCCCC"
STICKY_PAIRS = [(i, 22 - i) for i in range(4, 10)]
STICKY_BREAKS = [9]

# T3: hairpin in chain A (1-10) plus detached unpaired chain B (11-19)
DETACHED_SEQUENCE = "GGGAAAACCC" + "TTTTTTTTT"
DETACHED_PAIRS = [(1, 10), (2, 9), (3, 8)]
DETACHED_BREAKS = [10]

# T4: two independent hairpins, one per chain (A = 1-10, B = 11-20)
TWO_HAIRPIN_SEQUENCE = "GGGAAAACCC" + "GGGAAAACCC"
TWO_HAIRPIN_PAIRS = [(1, 10), (2, 9), (3, 8), (11, 20), (12, 19), (13, 18)]
TWO_HAIRPIN_BREAKS = [10]

# T5: a helix whose 5' strand crosses the junction:
# chain A = 1-8 (pairs 5-8 x 17-20 with chain B), chain B = 9-20
SPLIT_STEM_SEQUENCE = "G" * 8 + "C" * 12
SPLIT_STEM_PAIRS = [
    (5, 20),
    (6, 19),
    (7, 18),
    (8, 17),
    (9, 16),
    (10, 15),
]
SPLIT_STEM_BREAKS = [8]


# T1: blunt-end duplex -- the two chain termini pair with each other but no
# hairpin exists; the junction must yield per-chain terminal single strands.
def test_blunt_end_duplex_has_no_spurious_hairpin():
    bpseq = build_bpseq(BLUNT_PAIRS, BLUNT_SEQUENCE, strand_breaks=BLUNT_BREAKS)
    stems, single_strands, hairpins, loops = bpseq.elements

    assert len(stems) == 1
    assert (stems[0].strand5p.first, stems[0].strand5p.last) == (1, 8)
    assert (stems[0].strand3p.first, stems[0].strand3p.last) == (9, 16)
    assert hairpins == []
    assert loops == []
    assert single_strand_flags(single_strands) == {
        (False, True, 8, 8),
        (True, False, 9, 9),
    }
    assert_strands_do_not_cross_break(
        stems, single_strands, hairpins, loops, BLUNT_BREAKS
    )


# T2: duplex with 5' tails -- the junction region (A's last stem base, B's
# 5' tail, B's first stem base) used to be reported as one spurious hairpin.
def test_sticky_end_duplex_has_no_spurious_hairpin():
    bpseq = build_bpseq(STICKY_PAIRS, STICKY_SEQUENCE, strand_breaks=STICKY_BREAKS)
    stems, single_strands, hairpins, loops = bpseq.elements

    assert len(stems) == 1
    assert (stems[0].strand5p.first, stems[0].strand5p.last) == (4, 9)
    assert (stems[0].strand3p.first, stems[0].strand3p.last) == (13, 18)
    assert hairpins == []
    assert loops == []
    assert single_strand_flags(single_strands) == {
        (True, False, 1, 4),
        (False, True, 9, 9),
        (True, False, 10, 13),
    }
    assert_strands_do_not_cross_break(
        stems, single_strands, hairpins, loops, STICKY_BREAKS
    )


# T3: a fully unpaired chain must not be absorbed into the preceding
# chain's 3' single strand.
def test_detached_unpaired_chain_is_not_absorbed():
    bpseq = build_bpseq(
        DETACHED_PAIRS, DETACHED_SEQUENCE, strand_breaks=DETACHED_BREAKS
    )
    stems, single_strands, hairpins, loops = bpseq.elements

    assert len(stems) == 1
    assert [(h.strand.first, h.strand.last) for h in hairpins] == [(3, 8)]
    assert loops == []
    # only chain A's stem-closed 3' terminus; chain B yields no element
    assert single_strand_flags(single_strands) == {(False, True, 10, 10)}
    assert_strands_do_not_cross_break(
        stems, single_strands, hairpins, loops, DETACHED_BREAKS
    )


# T4: two independent per-chain hairpins -- the phantom single strand that
# used to bridge A's stem to B's stem across the junction must be gone.
def test_independent_per_chain_hairpins_have_no_bridge():
    bpseq = build_bpseq(
        TWO_HAIRPIN_PAIRS, TWO_HAIRPIN_SEQUENCE, strand_breaks=TWO_HAIRPIN_BREAKS
    )
    stems, single_strands, hairpins, loops = bpseq.elements

    assert len(stems) == 2
    assert sorted((s.strand5p.first, s.strand3p.first) for s in stems) == [
        (1, 8),
        (11, 18),
    ]
    assert sorted((h.strand.first, h.strand.last) for h in hairpins) == [
        (3, 8),
        (13, 18),
    ]
    assert loops == []
    assert single_strand_flags(single_strands) == {
        (False, True, 10, 10),
        (True, False, 11, 11),
    }
    assert_strands_do_not_cross_break(
        stems, single_strands, hairpins, loops, TWO_HAIRPIN_BREAKS
    )


# T5: a helix whose 5' strand crosses the junction must be split into two
# stems; the intra-chain hairpin of chain B must survive.
def test_stem_split_across_chain_junction():
    bpseq = build_bpseq(
        SPLIT_STEM_PAIRS, SPLIT_STEM_SEQUENCE, strand_breaks=SPLIT_STEM_BREAKS
    )
    stems, single_strands, hairpins, loops = bpseq.elements

    stem_positions = sorted(
        (s.strand5p.first, s.strand5p.last, s.strand3p.first, s.strand3p.last)
        for s in stems
    )
    assert stem_positions == [(5, 8, 17, 20), (9, 10, 15, 16)]
    # legitimate intra-chain hairpin of chain B
    assert [(h.strand.first, h.strand.last) for h in hairpins] == [(10, 15)]
    assert loops == []
    assert single_strand_flags(single_strands) == {
        (True, False, 1, 5),
        (False, True, 8, 8),
        (True, False, 9, 9),
        (False, False, 16, 17),
    }
    assert_strands_do_not_cross_break(
        stems, single_strands, hairpins, loops, SPLIT_STEM_BREAKS
    )


# T6: backward-compatibility contract -- without strand_breaks the legacy
# concatenation semantics (including the spurious junction hairpin) are
# preserved.
def test_without_breaks_legacy_behaviour_is_preserved():
    bpseq = build_bpseq(BLUNT_PAIRS, BLUNT_SEQUENCE)
    stems, single_strands, hairpins, loops = bpseq.elements

    assert bpseq.strand_breaks is None
    assert len(stems) == 1
    assert [(h.strand.first, h.strand.last) for h in hairpins] == [(8, 9)]
    assert single_strands == []
    assert loops == []


# T7: MultiStrandDotBracket input carries strand boundaries; from_dotbracket
# must derive strand_breaks from them.
def test_from_dotbracket_multistrand_derives_breaks():
    multi = MultiStrandDotBracket.from_multiline_string(
        ">A\nGGGGGGGG\n((((((((\n>B\nCCCCCCCC\n))))))))"
    )
    bpseq = BpSeq.from_dotbracket(multi)

    assert bpseq.strand_breaks == [8]
    stems, single_strands, hairpins, loops = bpseq.elements
    assert len(stems) == 1
    assert hairpins == []
    assert single_strand_flags(single_strands) == {
        (False, True, 8, 8),
        (True, False, 9, 9),
    }


# T8: strand breaks survive the pseudoknot-free and isolated-pair
# transformations used by extract_secondary_structure.
def test_transformations_preserve_breaks():
    bpseq = build_bpseq(BLUNT_PAIRS, BLUNT_SEQUENCE, strand_breaks=BLUNT_BREAKS)

    pk_free = bpseq.without_pseudoknots()
    assert pk_free.strand_breaks == [8]
    stems, single_strands, hairpins, loops = pk_free.elements
    assert len(stems) == 1
    assert hairpins == []

    # a structure with an isolated pair so without_isolated really rebuilds
    sequence = "GGGAAAACC" + "CC"
    pairs = [(1, 9), (2, 8), (3, 7), (10, 11)]
    bpseq = build_bpseq(pairs, sequence, strand_breaks=[9])
    without_isolated = bpseq.without_isolated()
    assert without_isolated.strand_breaks == [9]
    assert without_isolated.entries[9].pair == 0
    assert without_isolated.entries[10].pair == 0
    stems, single_strands, hairpins, loops = without_isolated.elements
    assert [(h.strand.first, h.strand.last) for h in hairpins] == [(3, 7)]
    assert single_strand_flags(single_strands) == {(False, True, 9, 9)}


# T9: orjson must be able to serialize a BpSeq carrying strand breaks (the
# annotator JSON pipeline serializes the full dataclass __dict__).
def test_orjson_serialization_includes_strand_breaks():
    bpseq = build_bpseq(BLUNT_PAIRS, BLUNT_SEQUENCE, strand_breaks=BLUNT_BREAKS)
    bpseq.elements  # simulate realistic cached state

    structure2d = Structure2D(
        base_pairs=[],
        stackings=[],
        base_ribose_interactions=[],
        base_phosphate_interactions=[],
        other_interactions=[],
        bpseq=bpseq,
        bpseq_index={},
        dot_bracket=MultiStrandDotBracket.from_string(
            BLUNT_SEQUENCE, "(((((((())))))))"
        ),
        extended_dot_bracket="(((((((())))))))",
        stems=[],
        single_strands=[],
        hairpins=[],
        loops=[],
        pseudoknot_stems=[],
    )

    processed = copy.deepcopy(structure2d)
    data = orjson.loads(
        orjson.dumps(
            processed,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        )
    )
    assert data["bpseq"]["strand_breaks"] == [8]


# T10: real-world integration -- 6INQ has chains T and N with a single
# nucleotide each, paired across chains.  The stem is real; the junction
# "hairpin" is not.
def test_6inq_two_chain_pairing_has_no_spurious_hairpin():
    with open("tests/6INQ.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d,
        base_interactions.base_pairs,
        base_interactions.stackings,
        True,
    )

    assert mapping.bpseq.strand_breaks == [1]
    stems, single_strands, hairpins, loops = mapping.bpseq.elements

    assert len(stems) == 1
    assert (stems[0].strand5p.first, stems[0].strand5p.last) == (1, 1)
    assert (stems[0].strand3p.first, stems[0].strand3p.last) == (2, 2)
    assert hairpins == []
    assert loops == []
    assert single_strand_flags(single_strands) == {
        (False, True, 1, 1),
        (True, False, 2, 2),
    }


# T11: property-style guard over shipped multi-chain structures: no element
# strand may span more than one chain.
def assert_no_element_strand_spans_multiple_chains(mapping):
    index_to_residue = mapping.bpseq_index_to_residue_map
    stems, single_strands, hairpins, loops = mapping.bpseq.elements

    strands = []
    for stem in stems:
        strands += [stem.strand5p, stem.strand3p]
    strands += [ss.strand for ss in single_strands]
    strands += [hairpin.strand for hairpin in hairpins]
    for loop in loops:
        strands += loop.strands

    for strand in strands:
        chains = {
            index_to_residue[index].chain
            for index in range(strand.first, strand.last + 1)
            if index in index_to_residue
        }
        assert len(chains) <= 1, f"Strand {strand} spans chains {chains}"


def test_6inq_elements_respect_chain_boundaries():
    with open("tests/6INQ.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d,
        base_interactions.base_pairs,
        base_interactions.stackings,
        True,
    )
    assert_no_element_strand_spans_multiple_chains(mapping)


def test_1jjp_elements_respect_chain_boundaries():
    # 1JJP is an inter-locked two-chain quadruplex; currently it yields no
    # canonical stems, so this passes vacuously and guards future regressions.
    with open("tests/1JJP.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d,
        base_interactions.base_pairs,
        base_interactions.stackings,
        True,
    )
    assert_no_element_strand_spans_multiple_chains(mapping)


# T12: end-to-end through the synthetic 3D pipeline -- a two-chain blunt-end
# duplex built from explicit residues and base pairs (no geometry needed:
# Mapping2D3D resolves residues by label).
def _synthetic_residue(chain, number, letter):
    label = ResidueLabel(chain=chain, number=number, name="D" + letter)
    auth = ResidueAuth(chain=chain, number=number, icode=None, name="D" + letter)
    return Residue3D(
        label=label,
        auth=auth,
        model=1,
        one_letter_name=letter,
        atoms=(),
        nucleotide_hint=True,
    )


def test_synthetic_two_chain_duplex_via_extract_secondary_structure():
    residues = [_synthetic_residue("A", number, "G") for number in range(1, 9)]
    residues += [_synthetic_residue("B", number, "C") for number in range(1, 9)]
    structure3d = Structure3D(residues)

    base_pairs = [
        BasePair(
            Residue(
                ResidueLabel("A", number, "DG"), ResidueAuth("A", number, None, "DG")
            ),
            Residue(
                ResidueLabel("B", 9 - number, "DC"),
                ResidueAuth("B", 9 - number, None, "DC"),
            ),
            LeontisWesthof.cWW,
            None,
        )
        for number in range(1, 9)
    ]
    base_interactions = BaseInteractions(base_pairs, [], [], [], [])

    structure2d, mapping = extract_secondary_structure(
        structure3d, base_interactions, find_gaps=False
    )

    assert mapping.bpseq.strand_breaks == [8]
    assert structure2d.hairpins == []
    assert len(structure2d.stems) == 1
    assert single_strand_flags(structure2d.single_strands) == {
        (False, True, 8, 8),
        (True, False, 9, 9),
    }
    assert_no_element_strand_spans_multiple_chains(mapping)
