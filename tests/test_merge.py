import pandas as pd
from rapidfuzz.fuzz import ratio

from fuzzy_df.merge import fuzz_merge


def test_fuzz_merge_inner_join():
    left = pd.DataFrame({"id_left": [1, 2], "name_left": ["foo", "bar"]})
    right = pd.DataFrame(
        {"id_right": [3, 4, 5, 6], "name_right": ["baz", "bear", "fool", "food"]}
    )

    result = fuzz_merge(
        left, right, left_on="name_left", right_on="name_right", score_cutoff=70
    )

    expected = pd.DataFrame(
        {
            "id_right": [4, 5, 6],
            "name_right": ["bear", "fool", "food"],
            "id_left": [2, 1, 1],
            "name_left": ["bar", "foo", "foo"],
            "left_index": [1, 0, 0],
            "right_index": [1, 2, 3],
            "score": [85.714287, 85.714287, 85.714287],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_fuzz_merge_missing_on_columns():
    left = pd.DataFrame({"id_left": [1, 2], "name_left": ["foo", "bar"]})
    right = pd.DataFrame({"id_right": [3, 4], "name_right": ["baz", "bear"]})

    try:
        fuzz_merge(left, right)
    except ValueError as e:
        assert str(e) == "on, left_on and right_on are required"


def test_fuzz_merge_non_inner_join():
    left = pd.DataFrame({"id_left": [1, 2], "name_left": ["foo", "bar"]})
    right = pd.DataFrame({"id_right": [3, 4], "name_right": ["baz", "bear"]})

    try:
        fuzz_merge(left, right, how="left")
    except NotImplementedError as e:
        assert str(e) == "Non inner join is supported not supported yet"


def test_fuzz_merge_multiple_columns_not_supported():
    left = pd.DataFrame({"id_left": [1, 2], "name_left": ["foo", "bar"]})
    right = pd.DataFrame({"id_right": [3, 4], "name_right": ["baz", "bear"]})

    try:
        fuzz_merge(left, right, left_on=["name_left", "id_left"], right_on="name_right")
    except NotImplementedError as e:
        assert str(e) == "Multiple columns not supported yet"


def test_fuzz_merge_custom_scorer():

    left = pd.DataFrame({"id_left": [1, 2], "name_left": ["foo", "bar"]})
    right = pd.DataFrame(
        {"id_right": [3, 4, 5], "name_right": ["baz", "bear", "foobar"]}
    )

    result = fuzz_merge(
        left,
        right,
        left_on="name_left",
        right_on="name_right",
        scorer=ratio,
        score_cutoff=80,
    )

    expected = pd.DataFrame(
        {
            "id_right": [4],
            "name_right": ["bear"],
            "id_left": [2],
            "name_left": ["bar"],
            "left_index": [1],
            "right_index": [1],
            "score": [85.714287],
        }
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
