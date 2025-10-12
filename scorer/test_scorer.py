from scorer import run_score

def test_scorer(expected, target_path, model_path):

    result = run_score(target_path, model_path, 10000, 10, False)

    assert isinstance(result, dict), "Scorer should return a dictionary"
    assert "score" in result, "Returned dictionary should contain 'score' key"
    assert result["score"] == expected, (
        f"Expected score {expected}, but got {result['score']} in {target_path} vs {model_path}"
    )
    print(f"Passed {target_path} vs {model_path}")


def run_tests():
    test_scorer(0.85, "1a9nr.pdb", "1a9nr.pdb")
    test_scorer(0.7740712124910121, "1a9nr.pdb", "1a9nr_M1.pdb")
    test_scorer(0.8757496105194233, "1i6uD.pdb", "1i6uD_M1.pdb")