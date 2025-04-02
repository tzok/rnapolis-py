import csv
import os
import tempfile
from pathlib import Path

from rnapolis.adapter import main


def test_adapter_fr3d():
    """Test adapter with FR3D output for 184D structure."""
    # Get paths to test files
    test_dir = Path(__file__).parent
    cif_path = test_dir / "184D.cif"
    fr3d_path = test_dir / "184D-fr3d.txt"
    expected_csv_path = test_dir / "184D.csv"

    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Run the adapter
        import sys

        old_argv = sys.argv
        sys.argv = [
            "adapter",
            str(cif_path),
            "--external",
            str(fr3d_path),
            "--tool",
            "fr3d",
            "--csv",
            output_path,
        ]
        main()
        sys.argv = old_argv

        # Compare the output with the expected output
        with (
            open(output_path, "r") as f_output,
            open(expected_csv_path, "r") as f_expected,
        ):
            output_reader = csv.reader(f_output)
            expected_reader = csv.reader(f_expected)

            output_rows = list(output_reader)
            expected_rows = list(expected_reader)

            # Check header
            assert output_rows[0] == expected_rows[0], "CSV headers don't match"

            # Check content (ignoring order)
            output_content = set(tuple(row) for row in output_rows[1:])
            expected_content = set(tuple(row) for row in expected_rows[1:])

            # Check if all expected interactions are in the output
            missing = expected_content - output_content
            assert not missing, f"Missing interactions: {missing}"

            # Check if there are no extra interactions in the output
            extra = output_content - expected_content
            assert not extra, f"Extra interactions: {extra}"

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)
