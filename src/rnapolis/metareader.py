#! /usr/bin/env python
import argparse
from typing import IO, Dict, List

import orjson
import pandas as pd
from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataContainer
from rnapolis.util import handle_input_file


def convert_category(data: List[DataContainer], category_name: str) -> List[Dict]:
    """
    Convert a single mmCIF category into a list of dictionaries.

    Args:
        data (List[DataContainer]): Parsed mmCIF data blocks.
        category_name (str): Name of the mmCIF category to extract.

    Returns:
        List of rows for the selected category, each row represented as a dictionary mapping mmCIF attribute names to their values. Returns an empty list if the category is not present.
    """
    category = data[0].getObj(category_name)
    if category:
        return [
            dict(zip(category.getAttributeList(), row)) for row in category.getRowList()
        ]
    return []


def read_metadata(file: IO[str], categories: List[str]) -> Dict:
    """
    Read selected metadata categories from an mmCIF file.

    Args:
        file (IO[str]): Open file handle pointing to an mmCIF file.
        categories (List[str]): Names of mmCIF categories to extract (e.g. ``["struct"]``).

    Returns:
        Dict: Mapping from category name to a list of row dictionaries for that category.
    """
    adapter = IoAdapterPy()
    data = adapter.readFile(file.name)
    return {key: convert_category(data, key) for key in categories}


def list_metadata(file: IO[str]) -> List[str]:
    """
    List all metadata categories available in an mmCIF file.

    Args:
        file (IO[str]): Open file handle pointing to an mmCIF file.

    Returns:
        Names of all mmCIF categories found in the file.
    """
    adapter = IoAdapterPy()
    data = adapter.readFile(file.name)
    return data[0].getObjNameList()


def main():
    """Command-line entry point for the ``metareader`` tool.

    The script:

    - reads an mmCIF file,
    - optionally lists all available metadata categories,
    - extracts selected categories and prints them as JSON,
    - optionally writes each extracted category to a separate CSV file.
    """

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
    parser.add_argument(
        "--csv-directory",
        help="directory where to output CSV per each category",
    )
    args = parser.parse_args()

    file = handle_input_file(args.path)

    if args.list_categories:
        for name in list_metadata(file):
            print(name)
    else:
        result = read_metadata(file, args.category)
        print(orjson.dumps(result).decode("utf-8"))

        if args.csv_directory:
            for category in result:
                with open(f"{args.csv_directory}/{category}.csv", "w") as f:
                    df = pd.DataFrame(result[category])
                    df.to_csv(f, index=False)


if __name__ == "__main__":
    main()
