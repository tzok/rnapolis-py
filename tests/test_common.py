from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    BasePhosphate,
    BaseRibose,
    BpSeq,
    DotBracket,
    Interaction,
    OtherInteraction,
    Residue,
    ResidueAuth,
    ResidueLabel,
    Stacking,
)


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
        "basePairs",
        "stackings",
        "baseRiboseInteractions",
        "basePhosphateInteractions",
        "otherInteractions",
    }


def test_bpseq_from_dotbracket():
    expected = BpSeq.from_file("tests/1ET4-A.bpseq")
    actual = BpSeq.from_dotbracket(DotBracket.from_file(f"tests/1ET4-A.dbn"))
    assert expected == actual


def test_elements():
    bpseq = BpSeq.from_dotbracket(DotBracket.from_file("tests/1EHZ.dbn"))
    stems, single_strands, hairpins, loops = bpseq.elements
    assert len(stems) == 5
    assert len(single_strands) == 1
    assert len(hairpins) == 1
    assert len(loops) == 2


def test_pseudoknot_order_assignment():
    bpseq = BpSeq.from_file("tests/6EK0-L5-L8.bpseq")
    dot_bracket = bpseq.dot_bracket

    counter = Counter(dot_bracket.structure)
    assert counter["."] == 1185
    assert counter["("] == 1298
    assert counter["["] == 44
    assert counter["{"] == 17
    assert counter["<"] == 7
    assert counter["A"] == 4
    assert counter["B"] == 1
    assert counter["C"] == 1
    assert counter["D"] == 0

    bpseq_again = BpSeq.from_dotbracket(dot_bracket)
    assert bpseq == bpseq_again
