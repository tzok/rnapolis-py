#! /usr/bin/env python
import argparse
import itertools
from typing import IO, Dict, List

import orjson
from mmcif.io.IoAdapterPy import IoAdapterPy
from mmcif.io.PdbxReader import DataContainer

from rnapolis.common import BpSeq, DotBracket
from rnapolis.util import handle_input_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbn", help="path to DotBracket file")
    parser.add_argument("--bpseq", help="path to BpSeq file")
    args = parser.parse_args()

    if args.dbn:
        bpseq = BpSeq.from_dotbracket(DotBracket.from_file(args.dbn))
    elif args.bpseq:
        bpseq = BpSeq.from_file(args.bpseq)
    else:
        parser.print_help()
        return

    print(f"Full dot-bracket:\n{bpseq.dot_bracket}")
    stems, single_strands, hairpins, loops = bpseq.elements

    for element in itertools.chain(stems, single_strands, hairpins, loops):
        print(element)


if __name__ == "__main__":
    main()
