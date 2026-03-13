import string
from collections import Counter

import orjson
from hypothesis import given, settings
from hypothesis import strategies as st

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BpSeq,
    DotBracket,
    Entry,
    Interaction,
    LeontisWesthof,
    MultiStrandDotBracket,
    OtherInteraction,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Saenger,
    Stacking,
)
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


@given(st.from_type(ResidueLabel))
def test_rnapdbee_adapters_api_compliance_residue_label(obj):
    assert obj.__dict__.keys() == {"chain", "number", "name"}


@given(st.from_type(ResidueAuth))
def test_rnapdbee_adapters_api_compliance_residue_auth(obj):
    assert obj.__dict__.keys() == {"chain", "number", "icode", "name"}


@given(st.from_type(Residue))
def test_rnapdbee_adapters_api_compliance_residue(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    obj.chain
    obj.number
    obj.icode
    obj.name
    obj.molecule_type
    obj.full_name
    assert obj.__dict__.keys() == {"label", "auth"}


@given(st.from_type(Interaction))
def test_rnapdbee_adapters_api_compliance_interaction(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2"}


@given(st.from_type(BasePair))
def test_rnapdbee_adapters_api_compliance_base_pair(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    obj.lw.reverse
    if obj.saenger is not None:
        obj.saenger.is_canonical
    assert obj.__dict__.keys() == {"nt1", "nt2", "lw", "saenger"}


@given(st.from_type(Stacking))
def test_rnapdbee_adapters_api_compliance_stacking(obj):
    # explicitly use all properties to make sure they are not added to __dict__ as @cached_property
    if obj.topology is not None:
        obj.topology.reverse
    assert obj.__dict__.keys() == {"nt1", "nt2", "topology"}


@given(st.from_type(BaseRibose))
def test_rnapdbee_adapters_api_compliance_base_ribose(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2", "br"}


@given(st.from_type(BasePhosphate))
def test_rnapdbee_adapters_api_compliance_base_phosphate(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2", "bph"}


@given(st.from_type(OtherInteraction))
def test_rnapdbee_adapters_api_compliance_other(obj):
    assert obj.__dict__.keys() == {"nt1", "nt2"}


@given(st.from_type(BaseInteractions))
@settings(max_examples=10)
def test_rnapdbee_adapters_api_compliance_structure2d(obj):
    assert obj.__dict__.keys() >= {
        "base_pairs",
        "stackings",
        "base_ribose_interactions",
        "base_phosphate_interactions",
        "other_interactions",
    }


def test_bpseq_from_dotbracket():
    expected = BpSeq.from_file("tests/1ET4-A.bpseq")
    actual = BpSeq.from_dotbracket(DotBracket.from_file("tests/1ET4-A.dbn"))
    assert expected == actual


def test_elements():
    bpseq = BpSeq.from_dotbracket(DotBracket.from_file("tests/1EHZ.dbn"))
    stems, single_strands, hairpins, loops = bpseq.elements
    assert len(stems) == 5
    assert len(single_strands) == 1
    assert len(hairpins) == 1
    assert len(loops) == 2


def test_elements_four_way_junction_zero_linkers():
    """Test that a 4-way junction with 0 nt linkers is detected (PDB 9E9Q)."""
    sequence = "CUCGUCUAUCUUCUGCAGGCUGCUUACGGGAAACCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAG"
    structure = "(((.(((((((((((..((((((.(((((....)))))..))))))......)))(((((((.((......)))))))))(((....))))))))))))))"
    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    stems, single_strands, hairpins, loops = bpseq.elements
    assert len(stems) == 8
    assert len(single_strands) == 0
    assert len(hairpins) == 3
    assert len(loops) == 5

    # Find the 4-way junction (the loop with 4 strands)
    four_way = [loop for loop in loops if len(loop.strands) == 4]
    assert len(four_way) == 1
    junction = four_way[0]

    # Verify the junction strand positions
    strand_positions = [(s.first, s.last) for s in junction.strands]
    assert strand_positions == [(12, 13), (55, 56), (80, 81), (90, 91)]

    # Verify no degenerate 2-strand loops with only paired residues
    for loop in loops:
        if len(loop.strands) == 2:
            assert any(s.last - s.first + 1 > 2 for s in loop.strands)


def test_no_degenerate_two_strand_loop():
    """Two directly stacking stems should not produce a spurious 2-strand loop."""
    # Use the same 9E9Q structure which previously produced a degenerate
    # Loop 64 65 GG (( 72 73 CC )) — two paired residues on each side,
    # no unpaired nucleotides, just re-describing a stem boundary.
    sequence = "CUCGUCUAUCUUCUGCAGGCUGCUUACGGGAAACCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAG"
    structure = "(((.(((((((((((..((((((.(((((....)))))..))))))......)))(((((((.((......)))))))))(((....))))))))))))))"

    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    stems, single_strands, hairpins, loops = bpseq.elements

    # No loop should consist of exactly 2 strands where both strands are
    # only paired residues (length <= 2)
    for loop in loops:
        if len(loop.strands) == 2:
            assert any(s.last - s.first + 1 > 2 for s in loop.strands)


def test_pseudoknot_order_assignment():
    bpseq = BpSeq.from_file("tests/6EK0-L5-L8.bpseq")
    dot_bracket = bpseq.dot_bracket

    counter = Counter(dot_bracket.structure)
    assert counter["."] == 1185
    assert counter["("] == 1298
    assert counter["["] == 44
    assert counter["{"] == 18
    assert counter["<"] == 6
    assert counter["A"] == 3
    assert counter["B"] == 2
    assert counter["C"] == 1
    assert counter["D"] == 0

    bpseq_again = BpSeq.from_dotbracket(dot_bracket)
    assert bpseq == bpseq_again


def test_multi_strand_dot_bracket():
    input = ">strand_A\nAGCGCCUGGACUUAAAGCCAU\n..((((.((((((((((((..\n>strand_B\nGGCUUUAAGUUGACGAGGGCAGGGUUUAUCGAGACAUCGGCGGGUGCCCUGCGGUCUUCCUGCGACCGUUAGAGGACUGGUAAAACCACAGGCGACUGUGGCAUAGAGCAGUCCGGGCAGGAA\n)))))))))))..(((...[[[[[[...)))......)))))...]]]]]][[[[[.((((((]]]]].....((((((......((((((....)))))).......))))))..))))))."
    dot_bracket = MultiStrandDotBracket.from_multiline_string(input)
    assert len(dot_bracket.strands) == 2
    assert dot_bracket.strands[0].sequence == "AGCGCCUGGACUUAAAGCCAU"
    assert dot_bracket.strands[1].sequence == (
        "GGCUUUAAGUUGACGAGGGCAGGGUUUAUCGAGACAUCGGCGGGUGCCCUGCGGUCUUCCUGCGACCGUUAGAGGACUGGUAAAACCACAGGCGACUGUGGCAUAGAGCAGUCCGGGCAGGAA"
    )
    assert dot_bracket.strands[0].structure == ("..((((.((((((((((((..")
    assert dot_bracket.strands[1].structure == (
        ")))))))))))..(((...[[[[[[...)))......)))))...]]]]]][[[[[.((((((]]]]].....((((((......((((((....)))))).......))))))..))))))."
    )


def test_conflicted_base_pairs():
    with open("tests/1A1T_1_B-rnaview.json", "rb") as f:
        data = orjson.loads(f.read())

    base_pairs = []

    for obj in data.get("base_pairs", []):
        nt1 = Residue(
            None,
            ResidueAuth(
                obj["nt1"]["auth"]["chain"],
                obj["nt1"]["auth"]["number"],
                obj["nt1"]["auth"]["icode"],
                obj["nt1"]["auth"]["name"],
            ),
        )
        nt2 = Residue(
            None,
            ResidueAuth(
                obj["nt2"]["auth"]["chain"],
                obj["nt2"]["auth"]["number"],
                obj["nt2"]["auth"]["icode"],
                obj["nt2"]["auth"]["name"],
            ),
        )
        lw = LeontisWesthof(obj["lw"])
        saenger = Saenger(obj["saenger"]) if obj["saenger"] else None
        base_pairs.append(BasePair(nt1, nt2, lw, saenger))

    with open("tests/1A1T_1_B.cif") as f:
        structure3d = read_3d_structure(f)

    mapping = Mapping2D3D(structure3d, base_pairs, [], True)
    assert (
        mapping.dot_bracket == ">strand_B\nGGACUAGCGGAGGCUAGUCC\n((((((((....))))))))"
    )


def test_high_level_pseudoknot():
    entries = []
    brackets = "([{<" + string.ascii_uppercase

    for i in range(len(brackets)):
        entries.append(Entry(i + 1, "C", i + len(brackets) + 1))
        entries.append(Entry(i + len(brackets) + 1, "G", i + 1))

    bpseq = BpSeq(sorted(entries))
    dot_bracket = bpseq.fcfs
    assert dot_bracket.sequence == "C" * len(brackets) + "G" * len(brackets)
    assert (
        dot_bracket.structure
        == "([{<" + string.ascii_uppercase + ")]}>" + string.ascii_lowercase
    )


def test_bpseq_removal_options():
    sequence = (
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCCAUCCACAGAAUUCGCACCA"
    )
    structure = (
        "(((((((..((((....[[..)))).((((..(...)..)))).....(((((..]]...))))))))))))...."
    )

    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    assert bpseq.dot_bracket.sequence == sequence
    assert bpseq.dot_bracket.structure == structure

    bpseq_without_isolated = bpseq.without_isolated()
    assert bpseq_without_isolated.dot_bracket.sequence == sequence
    assert (
        bpseq_without_isolated.dot_bracket.structure
        == "(((((((..((((....[[..)))).((((.........)))).....(((((..]]...))))))))))))...."
    )

    bpseq_without_pseudoknots = bpseq.without_pseudoknots()
    assert bpseq_without_pseudoknots.dot_bracket.sequence == sequence
    assert (
        bpseq_without_pseudoknots.dot_bracket.structure
        == "(((((((..((((........)))).((((..(...)..)))).....(((((.......))))))))))))...."
    )

    bpseq_without_both = bpseq.without_isolated().without_pseudoknots()
    assert bpseq_without_both.dot_bracket.sequence == sequence
    assert (
        bpseq_without_both.dot_bracket.structure
        == "(((((((..((((........)))).((((.........)))).....(((((.......))))))))))))...."
    )


def test_compute_elements_without_pseudoknots_tRNA():
    """PK-free decomposition of 1EHZ tRNA retains PK chars in Strand.structure."""
    bpseq = BpSeq.from_dotbracket(DotBracket.from_file("tests/1EHZ.dbn"))
    full_db = bpseq.dot_bracket.structure

    # Full decomposition (with pseudoknot pairs counted as stems)
    full_stems, full_ss, full_hp, full_loops = bpseq.elements
    assert len(full_stems) == 5
    assert len(full_hp) == 1
    assert len(full_loops) == 2
    assert len(full_ss) == 1

    # PK-free decomposition with full dot-bracket override
    pk_free_bpseq = bpseq.without_pseudoknots()
    stems, single_strands, hairpins, loops = pk_free_bpseq.compute_elements(
        dotbracket_override=full_db
    )

    # PK-free should have fewer stems (the pseudoknotted stem is gone)
    assert len(stems) == 4
    # With the PK stem removed, what was a loop/junction in the full structure
    # may become hairpins in the PK-free decomposition
    assert len(single_strands) == 1
    assert len(hairpins) == 3
    assert len(loops) == 1

    # Verify that Strand.structure fields contain PK bracket characters
    all_structures = []
    for stem in stems:
        all_structures.append(stem.strand5p.structure)
        all_structures.append(stem.strand3p.structure)
    for hp in hairpins:
        all_structures.append(hp.strand.structure)
    for lp in loops:
        for strand in lp.strands:
            all_structures.append(strand.structure)
    for ss in single_strands:
        all_structures.append(ss.strand.structure)

    joined = "".join(all_structures)
    # The full dot-bracket for 1EHZ contains [ and ] pseudoknot chars
    assert "[" in full_db and "]" in full_db
    # At least some strand structures should contain PK characters
    assert "[" in joined or "]" in joined


def test_compute_elements_without_pseudoknots_no_pk():
    """When there are no pseudoknots, PK-free decomposition gives same results."""
    sequence = "GGGGAAAACCCC"
    structure = "((((....))))"
    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))

    full_stems, full_ss, full_hp, full_loops = bpseq.elements

    pk_free_bpseq = bpseq.without_pseudoknots()
    pk_stems, pk_ss, pk_hp, pk_loops = pk_free_bpseq.compute_elements(
        dotbracket_override=bpseq.dot_bracket.structure
    )

    assert len(pk_stems) == len(full_stems)
    assert len(pk_ss) == len(full_ss)
    assert len(pk_hp) == len(full_hp)
    assert len(pk_loops) == len(full_loops)


def test_compute_elements_without_pseudoknots_multiple_pk_orders():
    """6EK0-L5-L8 with multiple PK orders: all PK stems captured."""
    bpseq = BpSeq.from_file("tests/6EK0-L5-L8.bpseq")
    full_db = bpseq.dot_bracket.structure

    # Full decomposition
    full_stems, _, _, _ = bpseq.elements

    # PK-free decomposition
    pk_free_bpseq = bpseq.without_pseudoknots()
    pk_free_stems, _, _, _ = pk_free_bpseq.compute_elements(dotbracket_override=full_db)

    # There should be fewer stems in the PK-free version
    assert len(pk_free_stems) < len(full_stems)

    # Identify PK stems: those whose Strand.structure contains non-() characters
    canonical = set("()")
    pk_stems = [
        stem
        for stem in full_stems
        if any(c not in canonical for c in stem.strand5p.structure)
        or any(c not in canonical for c in stem.strand3p.structure)
    ]

    # There should be multiple PK stems given the structure has [, {, <, A, B, C
    assert len(pk_stems) >= 2


def test_pseudoknot_stems_retain_full_dotbracket():
    """Strand.structure in PK stems should contain actual PK bracket characters."""
    sequence = (
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCCAUCCACAGAAUUCGCACCA"
    )
    structure = (
        "(((((((..((((....[[..)))).((((..(...)..)))).....(((((..]]...))))))))))))...."
    )

    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    full_stems, _, _, _ = bpseq.elements

    # Find pseudoknot stems by checking for non-() characters in Strand.structure
    canonical = set("()")
    pk_stems = [
        stem
        for stem in full_stems
        if any(c not in canonical for c in stem.strand5p.structure)
        or any(c not in canonical for c in stem.strand3p.structure)
    ]

    assert len(pk_stems) == 1

    pk_stem = pk_stems[0]
    assert "[" in pk_stem.strand5p.structure
    assert "]" in pk_stem.strand3p.structure


def test_pseudoknot_stems_populated_in_default_mode():
    """pseudoknot_stems is populated even without decompose-pseudoknot-free."""
    # 1EHZ tRNA has one pseudoknot stem (the [ ] pair)
    bpseq = BpSeq.from_dotbracket(DotBracket.from_file("tests/1EHZ.dbn"))
    full_stems, _, _, _ = bpseq.elements

    canonical = set("()")
    pk_stems = [
        stem
        for stem in full_stems
        if any(c not in canonical for c in stem.strand5p.structure)
        or any(c not in canonical for c in stem.strand3p.structure)
    ]

    assert len(pk_stems) == 1
    assert "[" in pk_stems[0].strand5p.structure
    assert "]" in pk_stems[0].strand3p.structure


def test_pseudoknot_stems_empty_when_no_pseudoknots():
    """pseudoknot_stems is empty for a structure without pseudoknots."""
    sequence = "GGGGAAAACCCC"
    structure = "((((....))))"
    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    full_stems, _, _, _ = bpseq.elements

    canonical = set("()")
    pk_stems = [
        stem
        for stem in full_stems
        if any(c not in canonical for c in stem.strand5p.structure)
        or any(c not in canonical for c in stem.strand3p.structure)
    ]

    assert len(pk_stems) == 0
    """Structure2D.pseudoknot_stems serializes correctly to JSON."""
    import copy
    import orjson

    from rnapolis.common import (
        Hairpin,
        InterStemParameters,
        Loop,
        MultiStrandDotBracket,
        Residue,
        SingleStrand,
        Stem,
        Strand,
        Structure2D,
    )

    # Create a minimal Structure2D with a pseudoknot stem
    pk_stem = Stem(
        Strand(18, 19, "GU", "[["),
        Strand(57, 58, "AC", "]]"),
    )

    structure2d = Structure2D(
        base_pairs=[],
        stackings=[],
        base_ribose_interactions=[],
        base_phosphate_interactions=[],
        other_interactions=[],
        bpseq=BpSeq.from_dotbracket(DotBracket("ACGU", "(())")),
        bpseq_index={},
        dot_bracket=MultiStrandDotBracket.from_string("ACGU", "(())"),
        extended_dot_bracket="(())",
        stems=[],
        single_strands=[],
        hairpins=[],
        loops=[],
        pseudoknot_stems=[pk_stem],
        inter_stem_parameters=[],
    )

    # Serialize to JSON
    processed = copy.deepcopy(structure2d)
    processed.bpseq_index = {
        k: Residue(v.label, v.auth) for k, v in structure2d.bpseq_index.items()
    }
    data = orjson.loads(
        orjson.dumps(
            processed,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        )
    )

    assert "pseudoknot_stems" in data
    assert len(data["pseudoknot_stems"]) == 1
    pk = data["pseudoknot_stems"][0]
    assert pk["strand5p"]["first"] == 18
    assert pk["strand5p"]["last"] == 19
    assert pk["strand5p"]["structure"] == "[["
    assert pk["strand3p"]["first"] == 57
    assert pk["strand3p"]["last"] == 58
    assert pk["strand3p"]["structure"] == "]]"
