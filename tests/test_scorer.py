import os
import random

import pytest

from rnapolis.scorer import evaluate_similarity

DATA_DIR = os.path.dirname(__file__)

REPS = 10000
SEED = 42


@pytest.mark.parametrize(
    "target_file, model_file",
    [
        ("1a9nR.pdb", "1a9nR.pdb"),
        ("1a9nR.pdb", "1a9nR_M1.pdb"),
        ("1i6uD.pdb", "1i6uD_M1.pdb"),
    ],
)
def test_scorer_returns_valid_result(target_file, model_file):
    target_path = os.path.join(DATA_DIR, target_file)
    model_path = os.path.join(DATA_DIR, model_file)

    if not os.path.exists(target_path) or not os.path.exists(model_path):
        pytest.skip(f"Test data not found: {target_file} or {model_file}")

    random.seed(SEED)
    result = evaluate_similarity(target_path, model_path, REPS, False)

    assert isinstance(result, dict), "Scorer should return a dictionary"
    assert "score" in result, "Returned dictionary should contain 'score' key"
    assert isinstance(result["score"], float), "Score should be a float"
    assert (
        0 <= result["score"] <= 1
    ), f"Score {result['score']} should be between 0 and 1"


def test_scorer_identical_structures():
    """When target and model are the same, the score should be exactly 1.0."""
    target_path = os.path.join(DATA_DIR, "1a9nR.pdb")
    if not os.path.exists(target_path):
        pytest.skip("Test data not found: 1a9nR.pdb")

    random.seed(SEED)
    result = evaluate_similarity(target_path, target_path, REPS, False)
    # Identical structures must produce a perfect similarity score
    assert (
        result["score"] == 1.0
    ), f"Self-comparison score {result['score']} should be exactly 1.0"
