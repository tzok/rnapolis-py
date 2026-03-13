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
    assert len(loops) == 6

    # Find the 4-way junction (the loop with 4 strands)
    four_way = [loop for loop in loops if len(loop.strands) == 4]
    assert len(four_way) == 1
    junction = four_way[0]

    # Verify the junction strand positions
    strand_positions = [(s.first, s.last) for s in junction.strands]
    assert strand_positions == [(12, 13), (55, 56), (80, 81), (90, 91)]


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
    pk_free_stems, _, _, _ = pk_free_bpseq.compute_elements(
        dotbracket_override=full_db
    )

    # There should be fewer stems in the PK-free version
    assert len(pk_free_stems) < len(full_stems)

    # Identify PK stems by checking which full stems have no pairs in PK-free
    pk_free_paired = set()
    for entry in pk_free_bpseq.entries:
        if entry.pair != 0:
            pk_free_paired.add(
                (min(entry.index_, entry.pair), max(entry.index_, entry.pair))
            )

    pk_stem_count = 0
    for stem in full_stems:
        stem_in_pk_free = False
        for idx in range(stem.strand5p.first, stem.strand5p.last + 1):
            partner = bpseq.pairs.get(idx, 0)
            if partner != 0 and (min(idx, partner), max(idx, partner)) in pk_free_paired:
                stem_in_pk_free = True
                break
        if not stem_in_pk_free:
            pk_stem_count += 1

    # There should be multiple PK stems given the structure has [, {, <, A, B, C
    assert pk_stem_count > 1

    # Full stems = PK-free stems + PK stems (approximately; they come from
    # different decompositions so exact equality isn't guaranteed, but the
    # PK stems should be a non-trivial subset)
    assert pk_stem_count >= 2


def test_pseudoknot_stems_retain_full_dotbracket():
    """Strand.structure in PK stems should contain actual PK bracket characters."""
    sequence = (
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCCAUCCACAGAAUUCGCACCA"
    )
    structure = (
        "(((((((..((((....[[..)))).((((..(...)..)))).....(((((..]]...))))))))))))...."
    )

    bpseq = BpSeq.from_dotbracket(DotBracket(sequence, structure))
    full_db = bpseq.dot_bracket.structure
    pk_free_bpseq = bpseq.without_pseudoknots()

    full_stems, _, _, _ = bpseq.elements

    # Find the pseudoknot stem(s)
    pk_free_paired = set()
    for entry in pk_free_bpseq.entries:
        if entry.pair != 0:
            pk_free_paired.add(
                (min(entry.index_, entry.pair), max(entry.index_, entry.pair))
            )

    pk_stems = []
    for stem in full_stems:
        stem_in_pk_free = False
        for idx in range(stem.strand5p.first, stem.strand5p.last + 1):
            partner = bpseq.pairs.get(idx, 0)
            if partner != 0 and (min(idx, partner), max(idx, partner)) in pk_free_paired:
                stem_in_pk_free = True
                break
        if not stem_in_pk_free:
            pk_stems.append(stem)

    assert len(pk_stems) == 1

    pk_stem = pk_stems[0]
    # The 5' strand of the PK stem should contain "[" characters
    assert "[" in full_db[pk_stem.strand5p.first - 1 : pk_stem.strand5p.last]
    # The 3' strand should contain "]" characters
    assert "]" in full_db[pk_stem.strand3p.first - 1 : pk_stem.strand3p.last]

    # When we rebuild with full dot-bracket, the structure should reflect that
    from rnapolis.common import Stem, Strand

    rebuilt_stem = Stem(
        Strand(
            pk_stem.strand5p.first,
            pk_stem.strand5p.last,
            pk_stem.strand5p.sequence,
            full_db[pk_stem.strand5p.first - 1 : pk_stem.strand5p.last],
        ),
        Strand(
            pk_stem.strand3p.first,
            pk_stem.strand3p.last,
            pk_stem.strand3p.sequence,
            full_db[pk_stem.strand3p.first - 1 : pk_stem.strand3p.last],
        ),
    )
    assert "[" in rebuilt_stem.strand5p.structure
    assert "]" in rebuilt_stem.strand3p.structure


def test_structure2d_pseudoknot_stems_json_serialization():
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
        k: Residue(v.label, v.auth)
        for k, v in structure2d.bpseq_index.items()
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
