#!/usr/bin/env python3
import argparse
import os
import sys

from rnapolis.parser import is_cif
from rnapolis.parser_v2 import (
    fit_to_pdb,
    parse_cif_atoms,
    parse_pdb_atoms,
    write_cif,
    write_pdb,
)


def main():
    """Main function to run the splitter tool."""
    parser = argparse.ArgumentParser(
        description="Split a multi-model PDB or mmCIF file into separate files per model."
    )
    parser.add_argument("--output", "-o", help="Output directory", required=True)
    parser.add_argument(
        "--format",
        "-f",
        help="Output format (possible values: PDB, mmCIF, keep. Default: keep)",
        default="keep",
    )
    parser.add_argument("file", help="Input PDB or mmCIF file to split")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.file):
        print(f"Error: Input file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Read and parse the input file
    input_format = "mmCIF"
    try:
        with open(args.file) as f:
            if is_cif(f):
                atoms_df = parse_cif_atoms(f)
                model_column = "pdbx_PDB_model_num"
            else:
                atoms_df = parse_pdb_atoms(f)
                input_format = "PDB"
                model_column = "model"
    except Exception as e:
        print(f"Error parsing file {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    if atoms_df.empty:
        print(f"Warning: No atoms found in {args.file}", file=sys.stderr)
        sys.exit(0)

    # Check if model column exists
    if model_column not in atoms_df.columns:
        print(
            f"Error: Model column '{model_column}' not found in the parsed data from {args.file}.",
            file=sys.stderr,
        )
        print(
            "This might indicate an issue with the input file or the parser.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output format
    output_format = args.format.upper()
    if output_format == "KEEP":
        output_format = input_format
    elif output_format not in ["PDB", "MMCIF"]:
        print(
            f"Error: Invalid output format '{args.format}'. Choose PDB, mmCIF, or keep.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Group by model number
    grouped_by_model = atoms_df.groupby(model_column)

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    # Write each model to a separate file
    for model_num, model_df in grouped_by_model:
        # Ensure model_df is a DataFrame copy to avoid SettingWithCopyWarning
        model_df = model_df.copy()

        # Set the correct format attribute for the writer function
        model_df.attrs["format"] = input_format

        # Construct output filename
        ext = ".pdb" if output_format == "PDB" else ".cif"
        output_filename = f"{base_name}_model_{model_num}{ext}"
        output_path = os.path.join(args.output, output_filename)

        print(f"Writing model {model_num} to {output_path}...")

        try:
            if output_format == "PDB":
                df_to_write = fit_to_pdb(model_df)
                write_pdb(df_to_write, output_path)
            else:  # mmCIF
                write_cif(model_df, output_path)
        except ValueError as e:
            # Handle errors specifically from fit_to_pdb
            print(
                f"Error fitting model {model_num} from {args.file} to PDB: {e}. Skipping model.",
                file=sys.stderr,
            )
            continue
        except Exception as e:
            # Handle general writing errors
            print(
                f"Error writing file {output_path} for model {model_num}: {e}",
                file=sys.stderr,
            )
            # Optionally continue to next model or exit
            # sys.exit(1)

    print("Splitting complete.")


if __name__ == "__main__":
    main()
