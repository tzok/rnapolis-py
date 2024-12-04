#! /usr/bin/env python
import argparse
import string
import tempfile
from typing import Dict, Tuple

from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory


def copy_from_to(
    file_content: str,
    category: str = "atom_site",
    copy_from: str = "label_asym_id",
    copy_to: str = "auth_asym_id",
) -> str:
    adapter = IoAdapterPy()

    with tempfile.NamedTemporaryFile(mode="wt") as f:
        f.write(file_content)
        f.seek(0)
        data = adapter.readFile(f.name)

    if len(data) == 0 or category not in data[0].getObjNameList():
        return file_content

    category_obj = data[0].getObj(category)
    attributes = category_obj.getAttributeList()

    if copy_from not in attributes:
        return file_content

    transformed = []

    if copy_to not in attributes:
        attributes.append(copy_to)

    for row in category_obj.getRowList():
        i = attributes.index(copy_from)
        j = attributes.index(copy_to)
        if j >= len(row):
            row.append(row[i])
        else:
            row[j] = row[i]
        transformed.append(row)

    data[0].replace(DataCategory(category_obj, attributes, transformed))

    with tempfile.NamedTemporaryFile(mode="rt+") as f:
        adapter.writeFile(f.name, data)
        f.seek(0)
        return f.read()


def replace_value(
    file_content: str,
    category: str = "atom_site",
    column: str = "auth_asym_id",
    values: str = "".join([c for c in string.printable if c not in string.whitespace]),
) -> Tuple[str, Dict]:
    adapter = IoAdapterPy()
    with tempfile.NamedTemporaryFile(mode="wt") as f:
        f.write(file_content)
        f.seek(0)
        data = adapter.readFile(f.name)

    if len(data) == 0 or category not in data[0].getObjNameList():
        return file_content, {}

    category_obj = data[0].getObj(category)
    attributes = category_obj.getAttributeList()

    if column not in attributes:
        return file_content, {}

    transformed = []
    mapping = {}

    for row in category_obj.getRowList():
        i = attributes.index(column)

        if row[i] not in mapping:
            mapping[row[i]] = values[len(mapping)]

        row[i] = mapping[row[i]]
        transformed.append(row)

    data[0].replace(DataCategory(category_obj, attributes, transformed))

    with tempfile.NamedTemporaryFile(mode="rt+") as f:
        adapter.writeFile(f.name, data)
        f.seek(0)
        return f.read(), mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input mmCIF file")
    parser.add_argument("output", help="path to output mmCIF file")
    parser.add_argument(
        "--category", help="name of the category to work on, e.g., atom_site"
    )
    parser.add_argument(
        "--copy-from",
        help="name of a data item to copy from, e.g., label_asym_id (exclusive with --replace)",
    )
    parser.add_argument(
        "--copy-to",
        help="name of a data item to copy to, e.g., auth_asym_id (exclusive with --replace)",
    )
    parser.add_argument(
        "--replace",
        help="name of a data item to replace values, e.g., auth_asym_id (exclusive with --copy-from and --copy-to)",
    )
    parser.add_argument(
        "--values",
        help="values to replace with, e.g., ABCDEFGHIJKLMNOPQRSTUVWXYZ (exclusive with --copy-from and --copy-to)",
    )
    args = parser.parse_args()

    if args.copy_from and args.copy_to:
        output = copy_from_to(args.input, args.category, args.copy_from, args.copy_to)
    elif args.replace and args.values:
        output = replace_value(args.input, args.category, args.replace, args.values)
    else:
        parser.print_help()
        return

    with open(args.output, "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
