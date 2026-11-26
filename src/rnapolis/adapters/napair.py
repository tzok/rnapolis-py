import csv
import gzip
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Tuple

from rnapolis.common import (
    BaseInteractions,
    BasePair,
    LeontisWesthof,
    Residue,
    ResidueAuth,
    ResidueLabel,
)
from rnapolis.contracts import StructureLike


class InteractionType(Enum):
    BASE_PAIR = "base-pair"
    OTHER = "other"


# Per-residue identifier columns used by NAPAIR's tabular output
RESIDUE_FIELDS = (
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
)

# Columns identifying a NAPAIR row, shared by CSV, JSON, and JSON Lines output
SHARED_FIELDS = ("pdbid", "model", "family")

# Columns that must be present for a row to be parseable
PARSE_REQUIRED_COLUMNS = {
    "family",
    *(f"{field}{i}" for i in (1, 2) for field in RESIDUE_FIELDS),
}

# Strict Leontis-Westhof family notation, e.g. cWW, tHS, tSS
FAMILY_PATTERN = re.compile(r"^[ct][WHS][WHS]$", re.IGNORECASE)

_FORMAT_SUFFIXES = (".csv", ".json", ".jsonl", ".jsonnd")


def required_columns() -> set:
    """Column names identifying NAPAIR's tabular output (CSV header or row keys)."""
    return {
        *SHARED_FIELDS,
        *(f"{field}{i}" for i in (1, 2) for field in RESIDUE_FIELDS),
    }


def open_text(file_path: str):
    """Open a NAPAIR output file as text, transparently decompressing gzip."""
    if file_path.lower().endswith(".gz"):
        return gzip.open(file_path, "rt", encoding="utf-8", newline="")
    if file_path.lower().endswith(".zst"):
        raise ValueError(
            f"zstandard-compressed NAPAIR output is not supported: {file_path}"
        )
    return open(file_path, "r", encoding="utf-8", newline="")


def _clean(value: Any) -> Any:
    """Normalize the empty-value conventions of CSV and JSON output to None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in ("", "?", "."):
        return None
    return value


def _has_alt_or_symmetry(row: Mapping[str, Any]) -> bool:
    """Return True if any of the alternative location or symmetry fields are set."""
    fields = [
        f"{field}{i}"
        for i in (1, 2)
        for field in ("label_alt_id", "symmetry_operation")
    ]
    for field in fields:
        if _clean(row.get(field)) is not None:
            return True
    return False


def _classify_interaction(
    row: Mapping[str, Any],
) -> Tuple[InteractionType, LeontisWesthof | None]:
    """
    Classifies interaction based on the NAPAIR family (Leontis-Westhof notation).

    Returns:
        base-pair, LeontisWesthof
    """
    family = row.get("family")
    family = "" if family is None else str(family)
    if FAMILY_PATTERN.match(family):
        # Convert to the format expected by LeontisWesthof
        lw_format = f"{family[0].lower()}{family[1].upper()}{family[2].upper()}"
        return InteractionType.BASE_PAIR, LeontisWesthof[lw_format]
    logging.warning(f"NAPAIR unknown interaction from family: {family!r}")
    return InteractionType.OTHER, None


def _parse_residues(row: Mapping[str, Any]) -> Tuple[Residue, Residue]:
    """
    Parse NAPAIR row into 2 Residue objects.

    Both label and auth coordinates are used when available. In NAPAIR's JSON
    output, fields with null values are omitted entirely, so missing keys are
    treated like empty CSV fields.
    """
    residues = []

    for i in range(1, 3):
        label_chain = _clean(row.get(f"label_asym_id{i}"))
        label_number = _clean(row.get(f"label_seq_id{i}"))
        label_name = _clean(row.get(f"label_comp_id{i}"))
        auth_chain = _clean(row.get(f"auth_asym_id{i}"))
        auth_number = _clean(row.get(f"auth_seq_id{i}"))
        auth_name = _clean(row.get(f"auth_comp_id{i}"))
        insertion_code = _clean(row.get(f"pdbx_PDB_ins_code{i}"))

        label = None
        if label_number is not None and label_name is not None:
            label = ResidueLabel(label_chain, int(label_number), str(label_name))

        auth = None
        if auth_number is not None and auth_name is not None:
            auth = ResidueAuth(
                auth_chain,
                int(auth_number),
                insertion_code,
                str(auth_name),
            )
        residues.append(Residue(label, auth))

    return residues[0], residues[1]


def _format_suffix(file_path: str) -> str:
    """Output format of a NAPAIR file, inferred from its extension."""
    lowered = file_path.lower()
    if lowered.endswith(".gz"):
        lowered = lowered[: -len(".gz")]
    for suffix in _FORMAT_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    # Fall back to CSV, like previous versions of the parser
    return ".csv"


def _read_csv(file_path: str) -> List[Dict[str, Any]]:
    with open_text(file_path) as f:
        return list(csv.DictReader(f))


def _read_json(file_path: str) -> List[Dict[str, Any]]:
    """Read NAPAIR's JSON output: a mapping from pdbid to a list of rows."""
    with open_text(file_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [
            row
            for value in data.values()
            if isinstance(value, list)
            for row in value
            if isinstance(row, dict)
        ]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read NAPAIR's JSON Lines output: one row per line."""
    rows = []
    with open_text(file_path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                logging.warning(
                    f"Skipping invalid JSON line {line_number} in {file_path}: {error}"
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_rows(file_path: str) -> List[Dict[str, Any]]:
    """Read a NAPAIR output file (.csv, .json, .jsonl/.jsonnd) into row dicts."""
    suffix = _format_suffix(file_path)
    if suffix == ".json":
        return _read_json(file_path)
    if suffix in (".jsonl", ".jsonnd"):
        return _read_jsonl(file_path)
    return _read_csv(file_path)


def _validate_rows(file_path: str, rows: List[Dict[str, Any]]) -> None:
    """Ensure the rows carry the identifier columns the parser relies on."""
    if not rows:
        logging.warning(f"No rows found in NAPAIR file: {file_path}")
        return
    keys: set = set()
    for row in rows:
        keys.update(row.keys())
        if PARSE_REQUIRED_COLUMNS <= keys:
            return
    missing = sorted(PARSE_REQUIRED_COLUMNS - keys)
    raise ValueError(
        f"{file_path} does not look like NAPAIR output; "
        f"missing columns: {', '.join(missing)}"
    )


def _is_napair_row(row: Any) -> bool:
    """True if a JSON object has the NAPAIR row signature."""
    return isinstance(row, Mapping) and PARSE_REQUIRED_COLUMNS <= set(row.keys())


def is_napair_csv_file(file_path: str) -> bool:
    """True if a CSV file starts with NAPAIR's header."""
    try:
        with open_text(file_path) as f:
            header = next(csv.reader(f), [])
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return required_columns() <= set(header)


def is_napair_json_file(file_path: str) -> bool:
    """True if a JSON file contains NAPAIR's grouped output (pdbid to rows)."""
    try:
        with open_text(file_path) as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    if isinstance(data, dict):
        groups = [value for value in data.values() if isinstance(value, list) and value]
        if not groups:
            return False
        return _is_napair_row(groups[0][0])
    if isinstance(data, list) and data:
        return _is_napair_row(data[0])
    return False


def is_napair_jsonl_file(file_path: str) -> bool:
    """True if a JSON Lines file contains NAPAIR rows."""
    try:
        with open_text(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return _is_napair_row(json.loads(line))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return False


def is_napair_file(file_path: str) -> bool:
    """True if the file is one of NAPAIR's output formats (CSV, JSON, JSON Lines)."""
    return (
        is_napair_csv_file(file_path)
        or is_napair_json_file(file_path)
        or is_napair_jsonl_file(file_path)
    )


def parse_napair_output(
    file_paths: List[str], structure3d: StructureLike
) -> BaseInteractions:
    """
    Parse NAPAIR output files and convert to BaseInteractions.

    Supports the CSV, JSON, and JSON Lines formats produced by NAPAIR's
    assign_in_structure.py. Rows describing alternative locations or
    symmetry-generated residues are skipped. The model column is currently
    ignored.

    Args:
        file_paths: List of paths to NAPAIR output files containing basepair interactions

    Returns:
        BaseInteractions object containing the interactions found by NAPAIR
    """

    bp_interactions: list[BasePair] = []

    # Process each input file
    for file_path in file_paths:
        logging.info(f"Processing NAPAIR file: {file_path}")
        rows = _read_rows(file_path)
        _validate_rows(file_path, rows)
        for row_number, row in enumerate(rows, start=1):
            if _has_alt_or_symmetry(row):
                logging.warning(
                    f"Non-empty alt or symmetry_operation in {file_path} row: {row_number}"
                )
                continue

            r1, r2 = _parse_residues(row)
            interaction_type, interaction_subtype = _classify_interaction(row)
            if interaction_type == InteractionType.OTHER:
                continue
            assert interaction_subtype is not None
            interaction = BasePair(r1, r2, interaction_subtype, None)
            bp_interactions.append(interaction)

    return BaseInteractions.from_structure3d(  # pyright: ignore[reportUnknownMemberType]
        structure3d,
        bp_interactions,
        [],
        [],
        [],
        [],
    )
