from rnapolis.annotator import extract_base_interactions
from rnapolis.common import ResidueAuth, ResidueLabel
from rnapolis.parser import read_3d_structure
from rnapolis.tertiary import Mapping2D3D


# 2HY9 has a canonical quadruplex
def test_2HY9():
    with open("tests/2HY9.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )

    # tract 1
    g4 = structure3d.find_residue(ResidueLabel("A", 4, "DG"), None)
    g5 = structure3d.find_residue(ResidueLabel("A", 5, "DG"), None)
    g6 = structure3d.find_residue(ResidueLabel("A", 6, "DG"), None)

    # tract 2
    g10 = structure3d.find_residue(ResidueLabel("A", 10, "DG"), None)
    g11 = structure3d.find_residue(ResidueLabel("A", 11, "DG"), None)
    g12 = structure3d.find_residue(ResidueLabel("A", 12, "DG"), None)

    # tract 3
    g16 = structure3d.find_residue(ResidueLabel("A", 16, "DG"), None)
    g17 = structure3d.find_residue(ResidueLabel("A", 17, "DG"), None)
    g18 = structure3d.find_residue(ResidueLabel("A", 18, "DG"), None)

    # tract 4
    g22 = structure3d.find_residue(ResidueLabel("A", 22, "DG"), None)
    g23 = structure3d.find_residue(ResidueLabel("A", 23, "DG"), None)
    g24 = structure3d.find_residue(ResidueLabel("A", 24, "DG"), None)

    assert all(
        nt is not None
        for nt in [g4, g5, g6, g10, g11, g12, g16, g17, g18, g22, g23, g24]
    )

    # tetrad 1
    assert {g10, g22}.issubset(mapping.base_pair_graph[g4])  # type: ignore
    assert {g10, g22}.issubset(mapping.base_pair_graph[g18])  # type: ignore
    assert {g4, g18}.issubset(mapping.base_pair_graph[g10])  # type: ignore
    assert {g4, g18}.issubset(mapping.base_pair_graph[g22])  # type: ignore

    # tetrad 2
    assert {g11, g23}.issubset(mapping.base_pair_graph[g5])  # type: ignore
    assert {g11, g23}.issubset(mapping.base_pair_graph[g17])  # type: ignore
    assert {g5, g17}.issubset(mapping.base_pair_graph[g11])  # type: ignore
    assert {g5, g17}.issubset(mapping.base_pair_graph[g23])  # type: ignore

    # tetrad 3
    assert {g12, g24}.issubset(mapping.base_pair_graph[g6])  # type: ignore
    assert {g12, g24}.issubset(mapping.base_pair_graph[g16])  # type: ignore
    assert {g6, g16}.issubset(mapping.base_pair_graph[g12])  # type: ignore
    assert {g6, g16}.issubset(mapping.base_pair_graph[g24])  # type: ignore


# 6RS3 has a quadruplex made of modified residues (GF2)
def test_6RS3():
    with open("tests/6RS3.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )

    g1 = structure3d.find_residue(ResidueLabel("A", 1, "DG"), None)
    g2 = structure3d.find_residue(ResidueLabel("A", 2, "DG"), None)
    g6 = structure3d.find_residue(ResidueLabel("A", 6, "DG"), None)
    g7 = structure3d.find_residue(ResidueLabel("A", 7, "DG"), None)
    g8 = structure3d.find_residue(ResidueLabel("A", 8, "DG"), None)
    g14 = structure3d.find_residue(ResidueLabel("A", 14, "GF2"), None)
    g15 = structure3d.find_residue(ResidueLabel("A", 15, "GF2"), None)
    g16 = structure3d.find_residue(ResidueLabel("A", 16, "DG"), None)
    g17 = structure3d.find_residue(ResidueLabel("A", 17, "DG"), None)
    g20 = structure3d.find_residue(ResidueLabel("A", 20, "DG"), None)
    g21 = structure3d.find_residue(ResidueLabel("A", 21, "DG"), None)
    g22 = structure3d.find_residue(ResidueLabel("A", 22, "DG"), None)

    assert all(
        nt is not None for nt in [g1, g2, g6, g7, g8, g14, g15, g16, g17, g20, g21, g22]
    )

    # tetrad 1
    assert {g2, g20}.issubset(mapping.base_pair_graph[g6])  # type: ignore
    assert {g2, g20}.issubset(mapping.base_pair_graph[g15])  # type: ignore
    assert {g6, g15}.issubset(mapping.base_pair_graph[g2])  # type: ignore
    assert {g6, g15}.issubset(mapping.base_pair_graph[g20])  # type: ignore

    # tetrad 2
    assert {g1, g21}.issubset(mapping.base_pair_graph[g7])  # type: ignore
    assert {g1, g21}.issubset(mapping.base_pair_graph[g16])  # type: ignore
    assert {g7, g16}.issubset(mapping.base_pair_graph[g1])  # type: ignore
    assert {g7, g16}.issubset(mapping.base_pair_graph[g21])  # type: ignore

    # tetrad 3
    assert {g14, g22}.issubset(mapping.base_pair_graph[g8])  # type: ignore
    assert {g14, g22}.issubset(mapping.base_pair_graph[g17])  # type: ignore
    assert {g8, g17}.issubset(mapping.base_pair_graph[g14])  # type: ignore
    assert {g8, g17}.issubset(mapping.base_pair_graph[g22])  # type: ignore


# 1JJP has an inter-locked quadruplex made of two chains
def test_1JJP():
    with open("tests/1JJP.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )

    ag1 = structure3d.find_residue(ResidueLabel("A", 1, "DG"), None)
    ag2 = structure3d.find_residue(ResidueLabel("A", 2, "DG"), None)
    ag3 = structure3d.find_residue(ResidueLabel("A", 3, "DG"), None)
    ag5 = structure3d.find_residue(ResidueLabel("A", 5, "DG"), None)
    ag6 = structure3d.find_residue(ResidueLabel("A", 6, "DG"), None)
    ag10 = structure3d.find_residue(ResidueLabel("A", 10, "DG"), None)
    ag11 = structure3d.find_residue(ResidueLabel("A", 11, "DG"), None)
    ag12 = structure3d.find_residue(ResidueLabel("A", 12, "DG"), None)

    bg1 = structure3d.find_residue(ResidueLabel("B", 1, "DG"), None)
    bg2 = structure3d.find_residue(ResidueLabel("B", 2, "DG"), None)
    bg3 = structure3d.find_residue(ResidueLabel("B", 3, "DG"), None)
    bg5 = structure3d.find_residue(ResidueLabel("B", 5, "DG"), None)
    bg6 = structure3d.find_residue(ResidueLabel("B", 6, "DG"), None)
    bg10 = structure3d.find_residue(ResidueLabel("B", 10, "DG"), None)
    bg11 = structure3d.find_residue(ResidueLabel("B", 11, "DG"), None)
    bg12 = structure3d.find_residue(ResidueLabel("B", 12, "DG"), None)

    assert all(
        nt is not None
        for nt in [
            ag1,
            ag2,
            ag3,
            ag5,
            ag6,
            ag10,
            ag11,
            ag12,
            bg1,
            bg2,
            bg3,
            bg5,
            bg6,
            bg10,
            bg11,
            bg12,
        ]
    )

    # tetrad 1
    assert {ag6, ag12}.issubset(mapping.base_pair_graph[ag3])  # type: ignore
    assert {ag6, ag12}.issubset(mapping.base_pair_graph[ag10])  # type: ignore
    assert {ag3, ag10}.issubset(mapping.base_pair_graph[ag6])  # type: ignore
    assert {ag3, ag10}.issubset(mapping.base_pair_graph[ag12])  # type: ignore

    # tetrad 2
    assert {ag5, ag11}.issubset(mapping.base_pair_graph[bg1])  # type: ignore
    assert {ag5, ag11}.issubset(mapping.base_pair_graph[ag2])  # type: ignore
    assert {bg1, ag2}.issubset(mapping.base_pair_graph[ag5])  # type: ignore
    assert {bg1, ag2}.issubset(mapping.base_pair_graph[ag11])  # type: ignore

    # tetrad 3
    assert {bg5, bg11}.issubset(mapping.base_pair_graph[ag1])  # type: ignore
    assert {bg5, bg11}.issubset(mapping.base_pair_graph[bg2])  # type: ignore
    assert {ag1, bg2}.issubset(mapping.base_pair_graph[bg5])  # type: ignore
    assert {ag1, bg2}.issubset(mapping.base_pair_graph[bg11])  # type: ignore

    # tetrad 4
    assert {bg6, bg12}.issubset(mapping.base_pair_graph[bg3])  # type: ignore
    assert {bg6, bg12}.issubset(mapping.base_pair_graph[bg10])  # type: ignore
    assert {bg3, bg10}.issubset(mapping.base_pair_graph[bg6])  # type: ignore
    assert {bg3, bg10}.issubset(mapping.base_pair_graph[bg12])  # type: ignore


# 6FC9 has a quadruplex and a duplex
def test_6FC9():
    with open("tests/6FC9.cif") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )

    g1 = structure3d.find_residue(ResidueLabel("A", 1, "DG"), None)
    g2 = structure3d.find_residue(ResidueLabel("A", 2, "DG"), None)
    g5 = structure3d.find_residue(ResidueLabel("A", 5, "DG"), None)
    g6 = structure3d.find_residue(ResidueLabel("A", 6, "DG"), None)
    c7 = structure3d.find_residue(ResidueLabel("A", 7, "DC"), None)
    g8 = structure3d.find_residue(ResidueLabel("A", 8, "DG"), None)
    c9 = structure3d.find_residue(ResidueLabel("A", 9, "DC"), None)
    g10 = structure3d.find_residue(ResidueLabel("A", 10, "DG"), None)
    a11 = structure3d.find_residue(ResidueLabel("A", 11, "DA"), None)
    a12 = structure3d.find_residue(ResidueLabel("A", 12, "DA"), None)
    t16 = structure3d.find_residue(ResidueLabel("A", 16, "DT"), None)
    t17 = structure3d.find_residue(ResidueLabel("A", 17, "DT"), None)
    c18 = structure3d.find_residue(ResidueLabel("A", 18, "DC"), None)
    g19 = structure3d.find_residue(ResidueLabel("A", 19, "DG"), None)
    c20 = structure3d.find_residue(ResidueLabel("A", 20, "DC"), None)
    g21 = structure3d.find_residue(ResidueLabel("A", 21, "DG"), None)
    g22 = structure3d.find_residue(ResidueLabel("A", 22, "DG"), None)
    g23 = structure3d.find_residue(ResidueLabel("A", 23, "DG"), None)
    g26 = structure3d.find_residue(ResidueLabel("A", 26, "DG"), None)
    g27 = structure3d.find_residue(ResidueLabel("A", 27, "DG"), None)

    assert all(
        nt is not None
        for nt in [
            g1,
            g2,
            g5,
            g6,
            c7,
            g8,
            c9,
            g10,
            a11,
            a12,
            t16,
            t17,
            c18,
            g19,
            c20,
            g21,
            g22,
            g23,
            g26,
            g27,
        ]
    )

    # tetrad 1
    assert {g6, g27}.issubset(mapping.base_pair_graph[g1])  # type: ignore
    assert {g6, g27}.issubset(mapping.base_pair_graph[g22])  # type: ignore
    assert {g1, g22}.issubset(mapping.base_pair_graph[g6])  # type: ignore
    assert {g1, g22}.issubset(mapping.base_pair_graph[g27])  # type: ignore

    # tetrad 2
    assert {g5, g26}.issubset(mapping.base_pair_graph[g2])  # type: ignore
    assert {g5, g26}.issubset(mapping.base_pair_graph[g23])  # type: ignore
    assert {g2, g23}.issubset(mapping.base_pair_graph[g5])  # type: ignore
    assert {g2, g23}.issubset(mapping.base_pair_graph[g26])  # type: ignore

    assert mapping.base_pair_dict[(c7, g21)].is_canonical  # type: ignore
    assert mapping.base_pair_dict[(g8, c20)].is_canonical  # type: ignore
    assert mapping.base_pair_dict[(c9, g19)].is_canonical  # type: ignore
    assert mapping.base_pair_dict[(g10, c18)].is_canonical  # type: ignore
    assert mapping.base_pair_dict[(a11, t17)].is_canonical  # type: ignore
    assert mapping.base_pair_dict[(a12, t16)].is_canonical  # type: ignore


# q-ugg is a quadruplex from molecular dynamics
def test_UGG_md():
    with open("tests/q-ugg-5k-salt_400-500ns_frame1065.pdb") as f:
        structure3d = read_3d_structure(f, 1)
    base_interactions = extract_base_interactions(structure3d)
    mapping = Mapping2D3D(
        structure3d, base_interactions.basePairs, base_interactions.stackings, True
    )

    u1 = structure3d.find_residue(None, ResidueAuth(" ", 1, None, "U5"))
    g2 = structure3d.find_residue(None, ResidueAuth(" ", 2, None, "G"))
    g3 = structure3d.find_residue(None, ResidueAuth(" ", 3, None, "G"))
    u4 = structure3d.find_residue(None, ResidueAuth(" ", 4, None, "U"))
    g5 = structure3d.find_residue(None, ResidueAuth(" ", 5, None, "G"))
    g6 = structure3d.find_residue(None, ResidueAuth(" ", 6, None, "G"))

    u8 = structure3d.find_residue(None, ResidueAuth(" ", 8, None, "U5"))
    g9 = structure3d.find_residue(None, ResidueAuth(" ", 9, None, "G"))
    g10 = structure3d.find_residue(None, ResidueAuth(" ", 10, None, "G"))
    u11 = structure3d.find_residue(None, ResidueAuth(" ", 11, None, "U"))
    g12 = structure3d.find_residue(None, ResidueAuth(" ", 12, None, "G"))
    g13 = structure3d.find_residue(None, ResidueAuth(" ", 13, None, "G"))

    u15 = structure3d.find_residue(None, ResidueAuth(" ", 15, None, "U5"))
    g16 = structure3d.find_residue(None, ResidueAuth(" ", 16, None, "G"))
    g17 = structure3d.find_residue(None, ResidueAuth(" ", 17, None, "G"))
    u18 = structure3d.find_residue(None, ResidueAuth(" ", 18, None, "U"))
    g19 = structure3d.find_residue(None, ResidueAuth(" ", 19, None, "G"))
    g20 = structure3d.find_residue(None, ResidueAuth(" ", 20, None, "G"))

    u22 = structure3d.find_residue(None, ResidueAuth(" ", 22, None, "U5"))
    g23 = structure3d.find_residue(None, ResidueAuth(" ", 23, None, "G"))
    g24 = structure3d.find_residue(None, ResidueAuth(" ", 24, None, "G"))
    u25 = structure3d.find_residue(None, ResidueAuth(" ", 25, None, "U"))
    g26 = structure3d.find_residue(None, ResidueAuth(" ", 26, None, "G"))
    g27 = structure3d.find_residue(None, ResidueAuth(" ", 27, None, "G"))

    assert all(
        nt is not None
        for nt in [
            u1,
            g2,
            g3,
            u4,
            g5,
            g6,
            u8,
            g9,
            g10,
            u11,
            g12,
            g13,
            u15,
            g16,
            g17,
            u18,
            g19,
            g20,
            u22,
            g23,
            g24,
            u25,
            g26,
            g27,
        ]
    )

    # structure from MD have empty chain name
    assert u1.full_name == "U5/1"  # type: ignore
    assert g2.full_name == "G2"  # type: ignore

    # tetrad 1
    assert {u1, u15}.issubset(mapping.base_pair_graph[u8])  # type: ignore
    assert {u1, u15}.issubset(mapping.base_pair_graph[u22])  # type: ignore
    assert {u8, u22}.issubset(mapping.base_pair_graph[u1])  # type: ignore
    assert {u8, u22}.issubset(mapping.base_pair_graph[u15])  # type: ignore

    # tetrad 2
    assert {g2, g16}.issubset(mapping.base_pair_graph[g9])  # type: ignore
    assert {g2, g16}.issubset(mapping.base_pair_graph[g23])  # type: ignore
    assert {g9, g23}.issubset(mapping.base_pair_graph[g2])  # type: ignore
    assert {g9, g23}.issubset(mapping.base_pair_graph[g16])  # type: ignore

    # tetrad 3
    assert {g3, g17}.issubset(mapping.base_pair_graph[g10])  # type: ignore
    assert {g3, g17}.issubset(mapping.base_pair_graph[g24])  # type: ignore
    assert {g10, g24}.issubset(mapping.base_pair_graph[g3])  # type: ignore
    assert {g10, g24}.issubset(mapping.base_pair_graph[g17])  # type: ignore

    # tetrad 4
    assert {u4, u18}.issubset(mapping.base_pair_graph[u11])  # type: ignore
    assert {u4, u18}.issubset(mapping.base_pair_graph[u25])  # type: ignore
    assert {u11, u25}.issubset(mapping.base_pair_graph[u4])  # type: ignore
    assert {u11, u25}.issubset(mapping.base_pair_graph[u18])  # type: ignore

    # tetrad 5
    assert {g5, g19}.issubset(mapping.base_pair_graph[g12])  # type: ignore
    assert {g5, g19}.issubset(mapping.base_pair_graph[g26])  # type: ignore
    assert {g12, g26}.issubset(mapping.base_pair_graph[g5])  # type: ignore
    assert {g12, g26}.issubset(mapping.base_pair_graph[g19])  # type: ignore

    # tetrad 6
    assert {g6, g20}.issubset(mapping.base_pair_graph[g13])  # type: ignore
    assert {g6, g20}.issubset(mapping.base_pair_graph[g27])  # type: ignore
    assert {g13, g27}.issubset(mapping.base_pair_graph[g6])  # type: ignore
    assert {g13, g27}.issubset(mapping.base_pair_graph[g20])  # type: ignore
