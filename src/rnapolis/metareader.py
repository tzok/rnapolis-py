#! /usr/bin/env python
import argparse
import json
import math

from mmcif.io.IoAdapterPy import IoAdapterPy


def find_values(data, category_name, obj_name):
    category = data[0].getObj(category_name)

    if category:
        for row in category.getRowList():
            row_dict = dict(zip(category.getAttributeList(), row))
            obj = row_dict.get(obj_name, None)

            if obj:
                yield obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to mmCIF file")
    args = parser.parse_args()

    adapter = IoAdapterPy()
    data = adapter.readFile(args.path)
    result = {
        "method": None,
        "resolution": math.nan,
        "title": None,
        "depositionDate": None,
        "releaseDate": None,
        "revisionDate": None,
    }

    if data:
        result["method"] = next(find_values(data, "exptl", "method"), None)
        result["resolution"] = min(
            map(
                float,
                list(find_values(data, "refine", "ls_d_res_high"))
                + list(find_values(data, "em_3d_reconstruction", "resolution"))
                + [math.nan],
            )
        )
        result["title"] = next(find_values(data, "struct", "title"), None)
        result["depositionDate"] = next(
            find_values(data, "pdbx_database_status", "recvd_initial_deposition_date"),
            None,
        )
        revision_dates = sorted(
            find_values(data, "pdbx_audit_revision_history", "revision_date")
        )
        if revision_dates:
            result["releaseDate"] = revision_dates[0]
            result["revisionDate"] = revision_dates[-1]

    print(json.dumps(result))


if __name__ == "__main__":
    main()
