#! /usr/bin/env python
import argparse
import sys

from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataCategory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to input mmCIF file")
    parser.add_argument("output", help="path to output mmCIF file")
    parser.add_argument(
        "--category", help="name of the category to work on, e.g., atom_site"
    )
    parser.add_argument(
        "--copy-from", help="name of a data item to copy from, e.g., label_asym_id"
    )
    parser.add_argument(
        "--copy-to", help="name of a data item to copy to, e.g., auth_asym_id"
    )
    args = parser.parse_args()

    adapter = IoAdapterPy()
    data = adapter.readFile(args.input)

    if len(data) == 0:
        print("Empty mmCIF file", file=sys.stderr)
        sys.exit(1)

    if args.category not in data[0].getObjNameList():
        print(f"Failed to find {args.category} in the mmCIF file", file=sys.stderr)
        sys.exit(1)

    category = data[0].getObj(args.category)

    if (
        args.copy_from not in category.getAttributeList()
        or args.copy_to not in category.getAttributeList()
    ):
        print(
            f"Failed to find data item {args.copy_from} or {args.copy_to} in {args.category}",
            file=sys.stderr,
        )
        sys.exit(1)

    transformed = []

    for row in category.getRowList():
        i = category.getAttributeList().index(args.copy_from)
        j = category.getAttributeList().index(args.copy_to)
        row[j] = row[i]
        transformed.append(row)

    data[0].replace(
        DataCategory(args.category, category.getAttributeList(), transformed)
    )

    adapter.writeFile(args.output, data)


if __name__ == "__main__":
    main()
