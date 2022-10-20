#! /usr/bin/env python
import argparse
import sys

import orjson

from mmcif.io.IoAdapterPy import IoAdapterPy


def convert_category(data, category_name):
    category = data[0].getObj(category_name)
    if category:
        return [
            dict(zip(category.getAttributeList(), row)) for row in category.getRowList()
        ]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to mmCIF file")
    parser.add_argument(
        "--category",
        "-c",
        help="an mmCIF category to extract, you can provide as many as you want (default=struct)",
        action="append",
        default=["struct"],
    )
    parser.add_argument(
        "--list-categories",
        "-l",
        help="read the mmCIF file and list categories available inside",
        action="store_true",
    )
    args = parser.parse_args()

    adapter = IoAdapterPy()
    data = adapter.readFile(args.path)

    if args.list_categories:
        for name in data[0].getObjNameList():
            print(name)
        sys.exit()

    result = {key: convert_category(data, key) for key in args.category}

    print(orjson.dumps(result).decode("utf-8"))


if __name__ == "__main__":
    main()
