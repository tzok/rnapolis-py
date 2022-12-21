from rnapolis.metareader import list_metadata, read_metadata
from rnapolis.util import handle_input_file


def test_list_metadata():
    assert ["atom_site"] == list_metadata(handle_input_file("tests/1JJP.cif"))
    assert [
        "audit_author",
        "struct",
        "struct_keywords",
        "exptl",
        "citation",
        "citation_author",
        "entity_poly",
        "entity_poly_seq",
        "entity",
        "pdbx_poly_seq_scheme",
        "pdbx_nonpoly_scheme",
        "chem_comp",
        "struct_asym",
        "atom_sites",
        "atom_type",
        "entry",
        "pdbx_chain_remapping",
        "pdbx_entity_nonpoly",
        "pdbx_entity_remapping",
        "atom_site",
    ] == list_metadata(handle_input_file("tests/1ehz-assembly-1.cif"))


def test_read_metadata():
    result = read_metadata(handle_input_file("tests/1ehz-assembly-1.cif"), ["struct"])
    assert "struct" in result
    assert len(result["struct"]) > 0
    assert "title" in result["struct"][0]
    assert (
        result["struct"][0]["title"]
        == "The crystal structure of yeast phenylalanine tRNA at 1.93 A resolution"
    )
