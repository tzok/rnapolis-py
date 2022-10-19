#! /usr/bin/env python
import argparse

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
    args = parser.parse_args()

    adapter = IoAdapterPy()
    data = adapter.readFile(args.path)

    result = {
        key: convert_category(data, key)
        for key in [
            "citation",
            "citation_author",
            "em_3d_reconstruction",
            "exptl",
            "pdbx_audit_revision_history",
            "pdbx_database_status",
            "refine",
        ]
    }

    print(orjson.dumps(result).decode("utf-8"))


if __name__ == "__main__":
    main()
