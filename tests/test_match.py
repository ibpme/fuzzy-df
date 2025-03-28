import pandas as pd
from rapidfuzz.fuzz import ratio

from fuzzy_df.match import fuzz_match


def test_fuzz_match_basic():
    comp_left = pd.Series(["apple", "banana", "cherry"])
    comp_right = pd.Series(["apples", "bananas", "grape"])
    result = fuzz_match(comp_left, comp_right, score_cutoff=70)
    expected = pd.DataFrame(
        {
            "left_index": [0, 1],
            "right_index": [0, 1],
            "score": [90.90908813476562, 92.30769348144531],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_fuzz_match_no_cutoff():
    comp_left = pd.Series(["apple", "banana"])
    comp_right = pd.Series(["apples", "bananas"])
    result = fuzz_match(comp_left, comp_right)
    expected = pd.DataFrame(
        {
            "left_index": [0, 0, 1, 1],
            "right_index": [0, 1, 0, 1],
            "score": [90.90908813476562, 16.666666, 16.666666, 92.30769348144531],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_fuzz_match_no_matches():
    comp_left = pd.Series(["apple", "banana", "cherry"])
    comp_right = pd.Series(["grape", "orange", "melon"])
    result = fuzz_match(comp_left, comp_right, score_cutoff=70)
    expected = pd.DataFrame(columns=["left_index", "right_index", "score"]).astype(
        {"left_index": "int64", "right_index": "int64", "score": "float64"}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_fuzz_match_custom_score_column():
    comp_left = pd.Series(["apple", "banana", "cherry"])
    comp_right = pd.Series(["apples", "bananas", "grape"])
    result = fuzz_match(comp_left, comp_right, score_col="similarity", score_cutoff=70)
    expected = pd.DataFrame(
        {
            "left_index": [0, 1],
            "right_index": [0, 1],
            "similarity": [90.90908813476562, 92.30769348144531],
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_fuzz_match_high_score_cutoff():
    comp_left = pd.Series(["apple", "banana", "cherry"])
    comp_right = pd.Series(["apples", "bananas", "grape"])
    result = fuzz_match(comp_left, comp_right, score_cutoff=95)
    expected = pd.DataFrame(columns=["left_index", "right_index", "score"]).astype(
        {"left_index": "int64", "right_index": "int64", "score": "float64"}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_fuzz_match_custom_scorer():
    left = pd.Series(["foo", "bar"])
    right = pd.Series(["baz", "bear", "foobar"])

    result = fuzz_match(
        left,
        right,
        scorer=ratio,
        score_cutoff=80,
    )

    expected = pd.DataFrame(
        {
            "left_index": [1],
            "right_index": [1],
            "score": [85.714287],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
