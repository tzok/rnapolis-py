#! /usr/bin/env python
import argparse
import tempfile
from typing import List, Set, Tuple

from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory, DataContainer

from rnapolis.util import handle_input_file

# Source: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
ENTITY_POLY_TYPES = [
    "cyclic-pseudo-peptide",
    "other",
    "peptide nucleic acid",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
    "polypeptide(D)",
    "polypeptide(L)",
    "polyribonucleotide",
]

CATEGORIES_WITH_ENTITY_ID = [
    ("atom_site", "label_entity_id"),
    ("entity_keywords", "entity_id"),
    ("entity_name_com", "entity_id"),
    ("entity_name_sys", "entity_id"),
    ("entity_poly", "entity_id"),
    ("entity_src_gen", "entity_id"),
    ("entity_src_nat", "entity_id"),
    ("pdbx_branch_scheme", "entity_id"),
    ("pdbx_chain_remapping", "entity_id"),
    ("pdbx_construct", "entity_id"),
    ("pdbx_entity_assembly", "entity_id"),
    ("pdbx_entity_branch", "entity_id"),
    ("pdbx_entity_branch_descriptor", "entity_id"),
    ("pdbx_entity_branch_list", "entity_id"),
    ("pdbx_entity_func_bind_mode", "entity_id"),
    ("pdbx_entity_name", "entity_id"),
    ("pdbx_entity_nonpoly", "entity_id"),
    ("pdbx_entity_poly_domain", "entity_id"),
    ("pdbx_entity_poly_na_nonstandard", "entity_id"),
    ("pdbx_entity_poly_na_type", "entity_id"),
    ("pdbx_entity_poly_protein_class", "entity_id"),
    ("pdbx_entity_prod_protocol", "entity_id"),
    ("pdbx_entity_remapping", "entity_id"),
    ("pdbx_entity_src_gen_character", "entity_id"),
    ("pdbx_entity_src_gen_chrom", "entity_id"),
    ("pdbx_entity_src_gen_clone", "entity_id"),
    ("pdbx_entity_src_gen_express", "entity_id"),
    ("pdbx_entity_src_gen_fract", "entity_id"),
    ("pdbx_entity_src_gen_lysis", "entity_id"),
    ("pdbx_entity_src_gen_prod_digest", "entity_id"),
    ("pdbx_entity_src_gen_prod_other", "entity_id"),
    ("pdbx_entity_src_gen_prod_pcr", "entity_id"),
    ("pdbx_entity_src_gen_proteolysis", "entity_id"),
    ("pdbx_entity_src_gen_pure", "entity_id"),
    ("pdbx_entity_src_gen_refold", "entity_id"),
    ("pdbx_entity_src_syn", "entity_id"),
    ("pdbx_linked_entity_list", "entity_id"),
    ("pdbx_prerelease_seq", "entity_id"),
    ("pdbx_sifts_xref_db", "entity_id"),
    ("pdbx_sifts_xref_db_segments", "entity_id"),
    ("pdbx_struct_entity_inst", "entity_id"),
    ("struct_asym", "entity_id"),
    ("struct_ref", "entity_id"),
]

CATEGORIES_WITH_ASYM_ID = [
    ("pdbx_coordinate_model", "asym_id"),
    ("pdbx_distant_solvent_atoms", "label_asym_id"),
    ("pdbx_linked_entity_instance_list", "asym_id"),
    ("pdbx_poly_seq_scheme", "asym_id"),
    ("pdbx_sifts_unp_segments", "asym_id"),
    ("pdbx_struct_asym_gen", "asym_id"),
    ("pdbx_struct_ncs_virus_gen", "asym_id"),
    ("pdbx_struct_special_symmetry", "label_asym_id"),
    ("pdbx_unobs_or_zero_occ_atoms", "label_asym_id"),
    ("pdbx_unobs_or_zero_occ_residues", "label_asym_id"),
    ("refine_ls_restr_ncs", "pdbx_asym_id"),
    ("struct_biol_gen", "asym_id"),
]

CATEGORIES_WITH_AUTH_ASYM_ID = [
    ("atom_site_anisotrop", "pdbx_auth_asym_id"),
    ("pdbx_atom_site_aniso_tls", "auth_asym_id"),
    ("pdbx_entity_instance_feature", "auth_asym_id"),
    ("pdbx_feature_monomer", "auth_asym_id"),
    ("pdbx_missing_atom_nonpoly", "auth_asym_id"),
    ("pdbx_missing_atom_poly", "auth_asym_id"),
    ("pdbx_modification_feature", "auth_asym_id"),
    ("pdbx_refine_component", "auth_asym_id"),
    ("pdbx_remediation_atom_site_mapping", "auth_asym_id"),
    ("pdbx_rmch_outlier", "auth_asym_id"),
    ("pdbx_rms_devs_cov_by_monomer", "auth_asym_id"),
    ("pdbx_sequence_pattern", "auth_asym_id"),
    ("pdbx_solvent_atom_site_mapping", "auth_asym_id"),
    ("pdbx_stereochemistry", "auth_asym_id"),
    ("pdbx_struct_chem_comp_diagnostics", "pdb_strand_id"),
    ("pdbx_struct_chem_comp_feature", "pdb_strand_id"),
    ("pdbx_struct_group_components", "auth_asym_id"),
    ("pdbx_struct_mod_residue", "auth_asym_id"),
    ("pdbx_sugar_phosphate_geometry", "auth_asym_id"),
    ("pdbx_validate_chiral", "auth_asym_id"),
    ("pdbx_validate_main_chain_plane", "auth_asym_id"),
    ("pdbx_validate_planes", "auth_asym_id"),
    ("pdbx_validate_planes_atom", "auth_asym_id"),
    ("pdbx_validate_torsion", "auth_asym_id"),
    ("struct_mon_nucl", "auth_asym_id"),
    ("struct_mon_prot", "auth_asym_id"),
    ("struct_site_gen", "auth_asym_id"),
]


def select_ids(
    data: List[DataContainer],
    obj_name: str,
    tested_field_name: str,
    extracted_field_name: str,
    accepted_values: Set[str],
) -> Set[str]:
    obj = data[0].getObj(obj_name)
    ids = set()

    if obj:
        for row in obj.getRowList():
            row_dict = dict(zip(obj.getAttributeList(), row))

            if row_dict.get(tested_field_name, None) in accepted_values:
                ids.add(row_dict[extracted_field_name])

    return ids


def select_category_by_id(
    data: List[DataContainer],
    category: str,
    field_name: str,
    ids: List[str],
) -> Tuple[List[str], List[List[str]]]:
    obj = data[0].getObj(category)
    attributes = []
    rows = []

    if obj:
        attributes = obj.getAttributeList()

        for row in obj.getRowList():
            row_dict = dict(zip(obj.getAttributeList(), row))

            if row_dict.get(field_name, None) in ids:
                rows.append(row)

    return attributes, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        help="a type of molecule to select (default: polyribonucleotide)",
        action="append",
        default=["polyribonucleotide"],
        choices=ENTITY_POLY_TYPES,
    )
    parser.add_argument("path", help="path to a PDBx/mmCIF file")
    args = parser.parse_args()

    file = handle_input_file(args.path)
    adapter = IoAdapterPy()
    data = adapter.readFile(file.name)
    output = DataContainer("rnapolis")

    entity_ids = select_ids(data, "entity_poly", "type", "entity_id", set(args.type))
    asym_ids = select_ids(data, "struct_asym", "entity_id", "id", entity_ids)
    auth_asym_ids = select_ids(
        data, "atom_site", "label_asym_id", "auth_asym_id", asym_ids
    )

    for table, ids in (
        (CATEGORIES_WITH_ENTITY_ID, entity_ids),
        (CATEGORIES_WITH_ASYM_ID, asym_ids),
        (CATEGORIES_WITH_AUTH_ASYM_ID, auth_asym_ids),
    ):
        for category, field_name in table:
            attributes, rows = select_category_by_id(data, category, field_name, ids)

            if attributes and rows:
                obj = DataCategory(category, attributes, rows)
                output.append(obj)

    with tempfile.NamedTemporaryFile() as tmp:
        adapter.writeFile(tmp.name, [output])
        print(tmp.read().decode())


if __name__ == "__main__":
    main()
