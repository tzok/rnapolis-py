from pathlib import Path

from rnapolis.adapter import ExternalTool, process_external_tool_output
from rnapolis.parser import read_3d_structure
from rnapolis.util import handle_input_file


def test_adapter_fr3d():
    """Test adapter with FR3D output for 1A4D structure."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"
    fr3d_path = test_dir / "1A4D_1_A-B-basepair_detail.txt"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process external tool output
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [str(fr3d_path)],
        ExternalTool.FR3D,
        str(cif_path),
        find_gaps=False,
    )

    # Check the dot-bracket output
    expected_dot_bracket = """>strand_A
GGCCGAUGGUAGUGUGGGGUC
((((.......(((((((...
>strand_B
UCCCCAUGCGAGAGUAGGCC
..))))))).......))))"""

    assert mapping.dot_bracket == expected_dot_bracket


def test_adapter_bpnet():
    """Test adapter with BPNet output for 1A4D structure."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"
    bpnet_path = test_dir / "1A4D_1_A-B-input_basepair.json"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process external tool output
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [str(bpnet_path)],
        ExternalTool.BPNET,
        str(cif_path),
        find_gaps=False,
    )

    # Check the dot-bracket output
    expected_dot_bracket = """>strand_A
GGCCGAUGGUAGUGUGGGGUC
((((.......(((((((...
>strand_B
UCCCCAUGCGAGAGUAGGCC
..))))))).......))))"""

    assert mapping.dot_bracket == expected_dot_bracket


def test_adapter_rnaview():
    """Test adapter with RNAView output for 1A4D structure."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"
    rnaview_path = test_dir / "1A4D_1_A-B-input.cif.out"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process external tool output
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [str(rnaview_path)],
        ExternalTool.RNAVIEW,
        str(cif_path),
        find_gaps=False,
    )

    # Check the dot-bracket output
    expected_dot_bracket = """>strand_A
GGCCGAUGGUAGUGUGGGGUC
((((.......(((((((...
>strand_B
UCCCCAUGCGAGAGUAGGCC
..))))))).......))))"""

    assert mapping.dot_bracket == expected_dot_bracket


def test_adapter_maxit():
    """Test adapter with MAXIT output for 1A4D structure."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"
    maxit_path = test_dir / "1A4D_1_A-B-output.cif"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process external tool output
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [str(maxit_path)],
        ExternalTool.MAXIT,
        str(cif_path),
        find_gaps=False,
    )

    # Check the dot-bracket output
    expected_dot_bracket = """>strand_A
GGCCGAUGGUAGUGUGGGGUC
((((.......((((((((..
>strand_B
UCCCCAUGCGAGAGUAGGCC
.)))))))).......))))"""

    assert mapping.dot_bracket == expected_dot_bracket


def test_adapter_dssr():
    """Test adapter with DSSR output for 1A4D structure."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"
    dssr_path = test_dir / "1A4D_1_A-B-dssr.json"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process external tool output
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [str(dssr_path)],
        ExternalTool.DSSR,
        str(cif_path),
        find_gaps=False,
    )

    # Check the dot-bracket output
    expected_dot_bracket = """>strand_A
GGCCGAUGGUAGUGUGGGGUC
((((.......((((((((..
>strand_B
UCCCCAUGCGAGAGUAGGCC
.)))))))).......))))"""

    assert mapping.dot_bracket == expected_dot_bracket


def test_adapter_auto_detection():
    """Test that auto-detection works correctly for different file types."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"

    # Read 3D structure once
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Test auto-detection for each tool
    test_cases = [
        (test_dir / "1A4D_1_A-B-basepair_detail.txt", ExternalTool.FR3D),
        (test_dir / "1A4D_1_A-B-input_basepair.json", ExternalTool.BPNET),
        (test_dir / "1A4D_1_A-B-input.cif.out", ExternalTool.RNAVIEW),
        (test_dir / "1A4D_1_A-B-output.cif", ExternalTool.MAXIT),
        (test_dir / "1A4D_1_A-B-dssr.json", ExternalTool.DSSR),
    ]

    from rnapolis.adapter import auto_detect_tool

    for file_path, expected_tool in test_cases:
        detected_tool = auto_detect_tool([str(file_path)])
        assert detected_tool == expected_tool, (
            f"Auto-detection failed for {file_path}: expected {expected_tool}, got {detected_tool}"
        )


def test_adapter_empty_files_maxit():
    """Test adapter with empty external files (should default to MAXIT)."""
    test_dir = Path(__file__).parent
    cif_path = test_dir / "1A4D_1_A-B.cif"

    # Read 3D structure
    file = handle_input_file(str(cif_path))
    structure3d = read_3d_structure(file, None)

    # Process with empty external files (should use MAXIT with input file)
    structure2d, mapping = process_external_tool_output(
        structure3d,
        [],  # Empty external files
        ExternalTool.MAXIT,
        str(cif_path),
        find_gaps=False,
    )

    # Should still produce some output (even if empty interactions)
    assert mapping.dot_bracket is not None
    assert structure2d is not None
