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
        "basePairs",
        "stackings",
        "baseRiboseInteractions",
        "basePhosphateInteractions",
        "otherInteractions",
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


def test_multi_strand_dot_bracket():
    input = ">strand_A\nAGCGCCUGGACUUAAAGCCAU\n..((((.((((((((((((..\n>strand_B\nGGCUUUAAGUUGACGAGGGCAGGGUUUAUCGAGACAUCGGCGGGUGCCCUGCGGUCUUCCUGCGACCGUUAGAGGACUGGUAAAACCACAGGCGACUGUGGCAUAGAGCAGUCCGGGCAGGAA\n)))))))))))..(((...[[[[[[...)))......)))))...]]]]]][[[[[.((((((]]]]].....((((((......((((((....)))))).......))))))..))))))."
    dot_bracket = MultiStrandDotBracket.from_string(input)
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

    for obj in data.get("basePairs", []):
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
