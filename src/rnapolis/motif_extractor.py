#! /usr/bin/env python
import argparse
import itertools

from rnapolis.common import BpSeq, DotBracket


def main():
    """Command-line entry point for the ``motif_extractor`` tool.

    The script:

    - reads secondary structure in DotBracket or BPSEQ format,
    - optionally removes pseudoknots and/or isolated base pairs,
    - extracts basic RNA secondary-structure elements (stems, single strands,
      hairpins, loops),
    - and prints them to stdout.

    Useful for quick inspection or preprocessing of RNA secondary structures.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbn", help="path to DotBracket file")
    parser.add_argument("--bpseq", help="path to BpSeq file")
    parser.add_argument(
        "--decompose-pseudoknot-free",
        action="store_true",
        help="decompose elements from pseudoknot-free structure",
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

    if args.decompose_pseudoknot_free:
        full_dotbracket_str = bpseq.dot_bracket.structure
        pk_free_bpseq = bpseq.without_pseudoknots()
        stems, single_strands, hairpins, loops = pk_free_bpseq.compute_elements(
            dotbracket_override=full_dotbracket_str
        )
        # Pseudoknot stems: stems from the full structure whose Strand.structure
        # contains characters other than '(' and ')'
        canonical = set("()")
        full_stems, _, _, _ = bpseq.elements
        pseudoknot_stems = [
            stem
            for stem in full_stems
            if any(c not in canonical for c in stem.strand5p.structure)
            or any(c not in canonical for c in stem.strand3p.structure)
        ]
    else:
        stems, single_strands, hairpins, loops = bpseq.elements
        pseudoknot_stems = []

    print(f"Full dot-bracket:\n{bpseq.dot_bracket}")

    for element in itertools.chain(stems, single_strands, hairpins, loops):
        print(element)

    for stem in pseudoknot_stems:
        print(
            f"PseudoknotStem {stem.strand5p.first} {stem.strand5p.last} "
            f"{stem.strand5p.sequence} {stem.strand5p.structure} "
            f"{stem.strand3p.first} {stem.strand3p.last} "
            f"{stem.strand3p.sequence} {stem.strand3p.structure}"
        )


if __name__ == "__main__":
    main()
