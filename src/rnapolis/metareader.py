#! /usr/bin/env python
import argparse
from typing import Dict, List

import orjson
from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataContainer


def convert_category(data: List[DataContainer], category_name: str) -> List[Dict]:
    category = data[0].getObj(category_name)
    if category:
        return [
            dict(zip(category.getAttributeList(), row)) for row in category.getRowList()
        ]
    return []


def read_metadata(path: str, categories: List[str]) -> Dict:
    adapter = IoAdapterPy()
    data = adapter.readFile(path)
    return {key: convert_category(data, key) for key in categories}


def list_metadata(path: str) -> List[str]:
    adapter = IoAdapterPy()
    data = adapter.readFile(path)
    return data[0].getObjNameList()


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

    if args.list_categories:
        for name in list_metadata(args.path):
            print(name)
    else:
        result = read_metadata(args.path, args.category)
        print(orjson.dumps(result).decode("utf-8"))


if __name__ == "__main__":
    main()
