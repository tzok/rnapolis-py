#! /usr/bin/env python
import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import appdirs
import requests
import RNA

from rnapolis.common import BpSeq, DotBracket

COMBINED_CM = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
SEPARATE_CM = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.tar.gz"


class FASTA:
    header: str
    sequence: str

    def __init__(self, header: str, sequence: str):
        self.header = header
        self.sequence = sequence.upper().replace("T", "U")

    def __str__(self):
        return f">{self.header}\n{self.sequence}"


def parse_fasta(fasta_path: str) -> List[FASTA]:
    """
    Read FASTA entries from a file.

    Args:
        fasta_path (str): The path to the FASTA file.

    Returns:
        List[Fasta]: A list of FASTA objects representing the entries in the file.
    """
    with open(fasta_path) as f:
        content = f.read()

    entries = content.split(">")[1:]
    fastas = []

    for entry in entries:
        lines = entry.splitlines()
        header = lines[0]
        sequence = "".join(lines[1:])
        fastas.append(FASTA(header, sequence))

    return fastas


def ensure_cm(family: str = None):
    if not os.path.exists(appdirs.user_data_dir("rnapolis")):
        os.makedirs(appdirs.user_data_dir("rnapolis"))

    if family is None:
        cm_gz_path = appdirs.user_data_dir("rnapolis") + "/Rfam.cm.gz"
        cm_path = appdirs.user_data_dir("rnapolis") + "/Rfam.cm"

        if not os.path.exists(cm_gz_path):
            response = requests.get(COMBINED_CM)

            with open(cm_gz_path, "wb") as f:
                f.write(response.content)

        if not os.path.exists(cm_path):
            with gzip.open(cm_gz_path, "rb") as f_in, open(cm_path, "wb") as f_out:
                f_out.write(f_in.read())
    else:
        cm_gz_path = appdirs.user_data_dir("rnapolis") + "/Rfam.tar.gz"
        cm_path = appdirs.user_data_dir("rnapolis") + f"/{family}.cm"

        if not os.path.exists(cm_gz_path):
            response = requests.get(SEPARATE_CM)

            with open(cm_gz_path, "wb") as f:
                f.write(response.content)

        if not os.path.exists(cm_path):
            shutil.unpack_archive(cm_gz_path, appdirs.user_data_dir("rnapolis"))

            if not os.path.exists(cm_path):
                raise RuntimeError(
                    f"Failed to find covariance model for {family} from Rfam."
                )

    if any(
        not os.path.exists(cm_path + extension)
        for extension in [".i1m", ".i1i", ".i1p", ".i1f"]
    ):
        for extension in [".i1m", ".i1i", ".i1p", ".i1f"]:
            if os.path.exists(cm_path + extension):
                os.remove(cm_path + extension)
        try:
            subprocess.run(["cmpress", cm_path], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print("Failed to run cmpress", file=sys.stderr)
            print(e.stdout.decode(), file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
            raise e

    return cm_path


def analyze_cmsearch(cmsearch: str, fasta: FASTA, count: int = 1):
    result = []
    lines = cmsearch.splitlines()
    begins = [i for i, line in enumerate(lines) if line.startswith(">>")]

    for i, begin in enumerate(begins):
        nc_index, cs_index = None, None

        for j in range(begin, begins[i + 1] if i + 1 < len(begins) else len(lines)):
            if lines[j].endswith(" NC"):
                nc_index = j
            if lines[j].endswith(" CS"):
                cs_index = j

        assert len(lines[cs_index].split()) == 2

        structure = lines[cs_index]
        sequence = lines[cs_index + 3]

        match = re.match(r"\s*.+?\s+(\d+)\s+.+\s+(\d+)", sequence)
        assert match is not None, sequence
        first, last = int(match.group(1)), int(match.group(2))

        for i in range(len(structure)):
            if structure[i] != " ":
                break

        j = structure.find(" CS")
        while structure[j] == " ":
            j -= 1
        j += 1

        structure = structure[i:j]
        sequence = sequence[i:j].upper()

        # remove pairs which did not match to consensus
        if nc_index is not None:
            non_canonical = lines[nc_index][i:j]
            for match in re.finditer(r"[v?]", non_canonical):
                i = match.start()
                structure = structure[:i] + "." + structure[i + 1 :]

        # replace *[n]* placeholders
        while True:
            match = re.search(r"[<*]\[ *(\d+)\][*>]", sequence)

            if match is None:
                break

            i, j = match.start(), match.end()
            n = int(match.group(1))
            sequence = sequence[:i] + "." * n + sequence[j:]
            structure = structure[:i] + "." * n + structure[j:]

        # replace gaps
        while True:
            match = re.search(r"-+", sequence)

            if match is None:
                break

            i, j = match.start(), match.end()
            sequence = sequence[:i] + sequence[j:]
            structure = structure[:i] + structure[j:]

        assert len(sequence) == len(structure)

        if first > last:
            # https://en.wikipedia.org/wiki/Nucleic_acid_notation
            complementary = {
                "A": "U",
                "C": "G",
                "G": "C",
                "U": "A",
                "W": "W",
                "S": "S",
                "M": "K",
                "K": "M",
                "R": "Y",
                "Y": "R",
                "B": "V",
                "D": "H",
                "H": "D",
                "V": "B",
                "N": "N",
                ".": ".",
            }
            assert set(sequence) <= set(complementary.keys()), (
                set(sequence) - set(complementary.keys()),
                sequence,
            )
            sequence_comp = "".join([complementary[c] for c in sequence[::-1]])
            match = re.search(sequence_comp, fasta.sequence)
            assert match is not None, (sequence, fasta.sequence)
            sequence = match.group()
        else:
            match = re.search(sequence, fasta.sequence)
            assert match is not None, (sequence, fasta.sequence)
            sequence = match.group()

        assert len(sequence) == len(structure)

        i = fasta.sequence.find(sequence)
        assert i != -1

        structure = (
            "." * i + structure + "." * (len(fasta.sequence) - len(sequence) - i)
        )
        sequence = fasta.sequence

        assert len(sequence) == len(structure)
        assert len(sequence) == len(fasta.sequence)

        structure = (
            structure.replace(":", ".")
            .replace("-", ".")
            .replace("_", ".")
            .replace(",", ".")
            .replace("~", ".")
        )
        if set(structure) == {"."}:
            continue

        dot_bracket = DotBracket.from_string("N" * len(structure), structure)
        structure = BpSeq.from_dotbracket(dot_bracket).dot_bracket.structure
        result.append([sequence, structure])

        if len(result) >= count:
            break

    if result == []:
        result.append([fasta.sequence, "." * len(fasta.sequence)])

    return result


def generate_consensus_secondary_structure(
    fasta: FASTA,
    family: str = None,
    fold: bool = True,
    count: int = 1,
    no_rfam_defaults: bool = False,
    lock: threading.Lock = None,
):
    if shutil.which("cmpress") is None or shutil.which("cmsearch") is None:
        raise RuntimeError(
            "cmpress/cmsearch not found in PATH, please install Infernal first."
        )

    if lock is not None:
        lock.acquire()

    cm_path = ensure_cm(family)

    if lock is not None:
        lock.release()

    with tempfile.NamedTemporaryFile(suffix=".fa") as fin:
        fin.write(str(fasta).encode())
        fin.seek(0)

        try:
            command = ["cmsearch", "--notextw"]
            if not no_rfam_defaults:
                command += ["--nohmmonly", "--rfam", "--cut_ga"]
            command += [cm_path, fin.name]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print("Failed to run cmsearch", file=sys.stderr)
            print(e.stdout.decode(), file=sys.stderr)
            print(e.stderr.decode(), file=sys.stderr)
            raise e

    results = analyze_cmsearch(completed.stdout.decode(), fasta, count)

    if fold:
        for i in range(len(results)):
            RNAfold = RNA.fold_compound(results[i][0])
            RNAfold.hc_add_from_db(results[i][1])
            structure, _ = RNAfold.mfe()
            results[i][1] = structure

    return [
        f">{fasta.header}\n{sequence}\n{structure}" for sequence, structure in results
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate consensus secondary structure for a given sequence. IMPORTANT! You need to have Infernal software installed to use this script."
    )
    parser.add_argument(
        "sequence",
        type=str,
        help="an RNA sequence or a path to FASTA file, possibly containing multiple sequences",
    )
    parser.add_argument(
        "--family",
        type=str,
        help="(optional) name of the Rfam family to use, if not given, the whole Rfam will be checked for the given sequence",
    )
    parser.add_argument(
        "--no-fold",
        action="store_true",
        help="(optional) whether to disable folding of the consensus secondary structure by RNAfold with constraints",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="(optional) maximum number of consensus secondary structures to generate per sequence, default is 1",
    )
    parser.add_argument(
        "--no-rfam-defaults",
        action="store_true",
        help="Infernal will be run with Rfam defaults (cmsearch --nohmmonly --rfam --cut_ga CM FASTA), "
        + "but if this is set then Infernal will be run with global defaults (cmsearch CM FASTA)",
    )

    args = parser.parse_args()

    if os.path.exists(args.sequence):
        fastas = parse_fasta(args.sequence)
    else:
        fastas = [FASTA("header", args.sequence)]

    lock = threading.Lock()

    with ThreadPoolExecutor() as executor:
        all_results = executor.map(
            lambda fasta: generate_consensus_secondary_structure(
                fasta,
                args.family,
                not args.no_fold,
                args.count,
                args.no_rfam_defaults,
                lock,
            ),
            fastas,
        )
        for per_fasta_results in all_results:
            for result in per_fasta_results:
                print(result)


if __name__ == "__main__":
    main()
