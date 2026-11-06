from pathlib import Path

from rnapolis.annotator import extract_base_interactions
from rnapolis.parser import read_3d_structure
from rnapolis.stem_geometry import (
    calculate_all_inter_stem_parameters,
    generate_pymol_script,
)
from rnapolis.util import handle_input_file

TEST_DIR = Path(__file__).parent


def test_stem_geometry_is_not_part_of_core_pipeline():
    """Stem geometry remains usable as an isolated experimental library."""
    path = TEST_DIR / "6g90_1.cif"
    structure = read_3d_structure(handle_input_file(str(path)), None)
    base_interactions = extract_base_interactions(structure)
    structure2d, mapping = structure.extract_secondary_structure(base_interactions)

    assert not hasattr(structure2d, "inter_stem_parameters")

    params = calculate_all_inter_stem_parameters(mapping)
    assert params
    assert params[0].stem1_idx != params[0].stem2_idx

    pml = generate_pymol_script(mapping, structure2d.stems)
    assert "pseudoatom" in pml
    assert "cmd.load_cgo" in pml
