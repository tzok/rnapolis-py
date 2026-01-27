#! /usr/bin/env python
import argparse
import os
import tempfile
from collections import defaultdict, namedtuple
from typing import Iterable, List, Set, Tuple

from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory, DataContainer

from rnapolis.parser_v2 import parse_cif_atoms, write_pdb
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

Link = namedtuple(
    "Link", ["parent_category_id", "parent_name", "child_category_id", "child_name"]
)


def load_pdbx_item_linked_group_list() -> dict[str, set["Link"]]:
    """Load linked item groups from the PDBx/mmCIF dictionary.

    Returns:
        Mapping from parent category ID to a set of link definitions.
    """
    dictionary = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "mmcif_pdbx_v50.dic"
    )
    adapter = IoAdapterPy()
    data = adapter.readFile(dictionary)
    obj = data[0].getObj("pdbx_item_linked_group_list")
    links = defaultdict(set)

    if obj:
        for row in obj.getRowList():
            row_dict = dict(zip(obj.getAttributeList(), row))
            child_category_id = row_dict["child_category_id"]
            child_name = row_dict["child_name"].split(".")[1]
            parent_name = row_dict["parent_name"].split(".")[1]
            parent_category_id = row_dict["parent_category_id"]
            links[parent_category_id].add(
                Link(parent_category_id, parent_name, child_category_id, child_name)
            )

    return links


def select_ids(
    data: List[DataContainer],
    category: str,
    field_name_to_extract: str,
    field_name_to_check: str,
    accepted_values: Iterable[str],
) -> Set[str]:
    """Select IDs from a category based on a field matching accepted values.

    Args:
        data (List[DataContainer]): Parsed mmCIF data containers.
        category (str): Category name to query (e.g. ``"entity_poly"``).
        field_name_to_extract (str): Field whose values will be returned.
        field_name_to_check (str): Field used to filter rows.
        accepted_values (Iterable[str]): Allowed values for the check field.

    Returns:
        Set of extracted IDs matching the filter criteria.
    """
    obj = data[0].getObj(category)
    if not obj:
        return set()
    attributes = obj.getAttributeList()
    if field_name_to_check not in attributes or field_name_to_extract not in attributes:
        return set()
    index_to_check = attributes.index(field_name_to_check)
    index_to_extract = attributes.index(field_name_to_extract)
    return {
        row[index_to_extract]
        for row in obj.getRowList()
        if row[index_to_check] in accepted_values
    }


def select_category_by_id(
    data: List[DataContainer],
    category: str,
    field_name: str,
    ids: Iterable[str],
) -> Tuple[List[str], List[List[str]]]:
    """Filter rows of a category by ID field values.

    Args:
        data (List[DataContainer]): Parsed mmCIF data containers.
        category (str): Category name to query.
        field_name (str): Field used to filter rows.
        ids (Iterable[str]): Accepted ID values.

    Returns:
        Tuple of (attribute names, filtered rows).
    """
    obj = data[0].getObj(category)
    if not obj:
        return [], []
    attributes = obj.getAttributeList()
    if field_name not in attributes:
        return attributes, []
    index = attributes.index(field_name)
    return attributes, [row for row in obj.getRowList() if row[index] in ids]


def read_cif(file_content: str) -> DataContainer:
    """Parse PDBx/mmCIF text content into data containers using the mmCIF IO adapter.

    Args:
        file_content (str): Raw PDBx/mmCIF file content.

    Returns:
        DataContainer: Parsed data container representing the input file.
    """
    with tempfile.NamedTemporaryFile("rt+") as f:
        adapter = IoAdapterPy()
        f.write(file_content)
        f.seek(0)
        return adapter.readFile(f.name)


def filter_cif(
    data: list[DataContainer],
    entity_ids: list[str],
    asym_ids: list[str],
    auth_asym_ids: list[str],
    retain_categories: list[str],
) -> str:
    """Build a filtered PDBx/mmCIF text containing only selected entities and categories.

    Args:
        data: Parsed mmCIF data containers.
        entity_ids: Entity IDs to keep.
        asym_ids: Asymmetric unit IDs (``struct_asym.id`` / ``label_asym_id``) to keep.
        auth_asym_ids: Author chain IDs (``auth_asym_id``) to keep.
        retain_categories: Additional categories to copy unchanged into the output.

    Returns:
        str: Filtered PDBx/mmCIF file content.
    """
    links = load_pdbx_item_linked_group_list()
    categories_with_entity_id = [("entity", "id")] + [
        (link.child_category_id, link.child_name)
        for link in links["entity"]
        if link.parent_name == "id"
    ]
    categories_with_asym_id = [("struct_asym", "id")] + [
        (link.child_category_id, link.child_name)
        for link in links["struct_asym"]
        if link.parent_name == "id"
    ]
    categories_with_auth_asym_id = [("atom_site", "auth_asym_id")] + [
        (link.child_category_id, link.child_name)
        for link in links["atom_site"]
        if link.parent_name == "auth_asym_id"
    ]

    output = DataContainer("rnapolis")

    for table, ids in (
        (categories_with_entity_id, entity_ids),
        (categories_with_asym_id, asym_ids),
        (categories_with_auth_asym_id, auth_asym_ids),
    ):
        for category, field_name in table:
            attributes, rows = select_category_by_id(data, category, field_name, ids)

            if attributes and rows:
                obj = DataCategory(category, attributes, rows)
                output.append(obj)

    for category in retain_categories:
        obj = data[0].getObj(category)
        if obj:
            output.append(obj)

    with tempfile.NamedTemporaryFile("rt+") as tmp:
        adapter = IoAdapterPy()
        adapter.writeFile(tmp.name, [output])
        tmp.seek(0)
        return tmp.read()


def filter_by_poly_types(
    file_content: str,
    entity_poly_types: Iterable[str] = [
        "polyribonucleotide",
        "polydeoxyribonucleotide",
        "polydeoxyribonucleotide/polyribonucleotide hybrid",
    ],
    retain_categories: Iterable[str] = ["chem_comp"],
) -> str:
    """Filter a PDBx/mmCIF file to nucleic-acid entities with selected polymer types.

    By default this keeps only RNA, DNA and RNA/DNA hybrid entities based on the
    ``_entity_poly.type`` field.

    Args:
        file_content (str): Raw PDBx/mmCIF file content.
        entity_poly_types (Iterable[str]): Allowed values of ``_entity_poly.type``.
        retain_categories (Iterable[str]): Additional categories to keep verbatim.

    Returns:
        str: Filtered PDBx/mmCIF file content.
    """
    data = read_cif(file_content)
    entity_ids = select_ids(
        data, "entity_poly", "entity_id", "type", set(entity_poly_types)
    )
    asym_ids = select_ids(data, "struct_asym", "id", "entity_id", entity_ids)
    auth_asym_ids = select_ids(
        data, "atom_site", "auth_asym_id", "label_asym_id", asym_ids
    )
    return filter_cif(data, entity_ids, asym_ids, auth_asym_ids, retain_categories)


def filter_by_entity_ids(
    file_content: str,
    entity_ids: Iterable[str],
    retain_categories: Iterable[str] = ["chem_comp"],
) -> str:
    """Filter a PDBx/mmCIF file to a given set of entity IDs.

    Args:
        file_content (str): Raw PDBx/mmCIF file content.
        entity_ids (Iterable[str]): Entity IDs to include.
        retain_categories (Iterable[str]): Additional categories to keep verbatim.

    Returns:
        str: Filtered PDBx/mmCIF file content.
    """
    data = read_cif(file_content)
    entity_ids = set(entity_ids)
    asym_ids = select_ids(data, "struct_asym", "id", "entity_id", entity_ids)
    auth_asym_ids = select_ids(
        data, "atom_site", "auth_asym_id", "label_asym_id", asym_ids
    )
    return filter_cif(data, entity_ids, asym_ids, auth_asym_ids, retain_categories)


def filter_by_chains(
    file_content: str,
    chains: Iterable[str],
    retain_categories: Iterable[str] = ["chem_comp"],
) -> str:
    """Filter a PDBx/mmCIF file by chain IDs (``label_asym_id``).

    Args:
        file_content (str): Raw PDBx/mmCIF file content.
        chains (Iterable[str]): Chain IDs (``label_asym_id``) to select.
        retain_categories (Iterable[str]): Additional categories to keep verbatim.

    Returns:
        str: Filtered PDBx/mmCIF file content.
    """
    data = read_cif(file_content)
    asym_ids = set(chains)
    entity_ids = select_ids(data, "struct_asym", "entity_id", "id", asym_ids)
    auth_asym_ids = select_ids(
        data, "atom_site", "auth_asym_id", "label_asym_id", asym_ids
    )
    return filter_cif(data, entity_ids, asym_ids, auth_asym_ids, retain_categories)


def main():
    """Command-line entry point for the ``molecule_filter`` tool.

    The CLI:

    - reads a PDBx/mmCIF file,
    - filters its content by polymer types (e.g. RNA/DNA), entity IDs or chain IDs,
    - optionally retains extra metadata categories,
    - can convert the filtered structure to PDB format.

    This is useful for preparing structural data for downstream analyses by
    removing components that are not relevant for a given task.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter-by-poly-types",
        help=f"filter by entity poly types, possible values: {', '.join(ENTITY_POLY_TYPES)}",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--filter-by-entity-ids",
        help="filter by entity IDs, e.g. 1, 2, 3",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--filter-by-chains",
        help="filter by chain IDs (label_asym_id), e.g. A, B, C",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--retain-categories",
        help="categories to retain in the output file default: chem_comp",
        action="append",
        default=["chem_comp"],
    )
    parser.add_argument(
        "--pdb", help="change output format to PDB", action="store_true"
    )
    parser.add_argument("path", help="path to a PDBx/mmCIF file")
    args = parser.parse_args()

    file = handle_input_file(args.path)
    content = None

    if args.filter_by_poly_types:
        content = filter_by_poly_types(
            file.read(),
            entity_poly_types=args.filter_by_poly_types,
            retain_categories=args.retain_categories,
        )
    elif args.filter_by_entity_ids:
        content = filter_by_entity_ids(
            file.read(),
            entity_ids=args.filter_by_entity_ids,
            retain_categories=args.retain_categories,
        )
    elif args.filter_by_chains:
        content = filter_by_chains(
            file.read(),
            chains=args.filter_by_chains,
            retain_categories=args.retain_categories,
        )
    else:
        parser.print_help()

    if content:
        if args.pdb:
            df = parse_cif_atoms(content)
            print(write_pdb(df))
        else:
            print(content)


if __name__ == "__main__":
    main()
