#! /usr/bin/env python
import argparse
import itertools

from rnapolis.common import BpSeq, DotBracket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbn", help="path to DotBracket file")
    parser.add_argument("--bpseq", help="path to BpSeq file")
    parser.add_argument(
        "--remove-pseudoknots", action="store_true", help="remove pseudoknots"
    )
    parser.add_argument(
        "--remove-isolated", action="store_true", help="remove isolated base pairs"
    )
    args = parser.parse_args()

    if args.dbn:
        bpseq = BpSeq.from_dotbracket(DotBracket.from_file(args.dbn))
    elif args.bpseq:
        bpseq = BpSeq.from_file(args.bpseq)
    else:
        parser.print_help()
        return

    if args.remove_isolated:
        bpseq = bpseq.without_isolated()

    if args.remove_pseudoknots:
        bpseq = bpseq.without_pseudoknots()

    print(f"Full dot-bracket:\n{bpseq.dot_bracket}")
    stems, single_strands, hairpins, loops = bpseq.elements

    for element in itertools.chain(stems, single_strands, hairpins, loops):
        print(element)


if __name__ == "__main__":
    main()
