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
        assert str(e) == "Argument 'on' or ('left_on' and 'right_on') are required"


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
        fuzz_merge(
            left, right, left_on=["name_left", "id_left"], right_on=["name_right"]
        )
    except NotImplementedError as e:
        assert str(e) == "Multiple columns for this config is not supported yet"


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


def test_fuzz_merge_multiple_columns_combine_strategy():
    left = pd.DataFrame(
        {
            "id_left": [1, 2],
            "first_name": ["John", "Jane"],
            "last_name": ["Doe", "Smith"],
        }
    )
    right = pd.DataFrame(
        {
            "id_right": [3, 4],
            "full_name": ["John Doe", "Jane Smith"],
        }
    )

    result = fuzz_merge(
        left,
        right,
        left_on=["first_name", "last_name"],
        right_on=["full_name"],
        strategy="combine",
        score_cutoff=90,
    )

    expected = pd.DataFrame(
        {
            "id_right": [3, 4],
            "full_name": ["John Doe", "Jane Smith"],
            "id_left": [1, 2],
            "first_name": ["John", "Jane"],
            "last_name": ["Doe", "Smith"],
            "left_index": [0, 1],
            "right_index": [0, 1],
            "score": [100.0, 100.0],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_fuzz_merge_multiple_columns_mismatch():
    left = pd.DataFrame(
        {
            "id_left": [1, 2],
            "first_name": ["John", "Jane"],
            "last_name": ["Doe", "Smith"],
        }
    )
    right = pd.DataFrame(
        {
            "id_right": [3, 4],
            "full_name": ["Alice Johnson", "Bob Brown"],
        }
    )

    result = fuzz_merge(
        left,
        right,
        left_on=["first_name", "last_name"],
        right_on=["full_name"],
        strategy="combine",
        score_cutoff=90,
    )

    expected = pd.DataFrame(
        columns=[
            "id_right",
            "full_name",
            "id_left",
            "first_name",
            "last_name",
            "left_index",
            "right_index",
            "score",
        ],
    )
    expected["score"] = expected["score"].astype(float)
    expected["left_index"] = expected["left_index"].astype(int)
    expected["right_index"] = expected["right_index"].astype(int)
    expected["id_right"] = expected["id_right"].astype(int)
    expected["id_left"] = expected["id_left"].astype(int)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_fuzz_merge_multiple_columns_invalid_strategy():
    left = pd.DataFrame(
        {
            "id_left": [1, 2],
            "first_name": ["John", "Jane"],
            "last_name": ["Doe", "Smith"],
        }
    )
    right = pd.DataFrame(
        {
            "id_right": [3, 4],
            "full_name": ["John Doe", "Jane Smith"],
        }
    )

    try:
        fuzz_merge(
            left,
            right,
            left_on=["first_name", "last_name"],
            right_on=["full_name"],
            strategy="unsupported_strategy",
        )
    except NotImplementedError as e:
        assert str(e) == "Multiple columns for this config is not supported yet"


def test_fuzz_merge_multiple_columns_one_side_list_error():
    left = pd.DataFrame(
        {
            "id_left": [1, 2],
            "first_name": ["John", "Jane"],
            "last_name": ["Doe", "Smith"],
        }
    )
    right = pd.DataFrame(
        {
            "id_right": [3, 4],
            "full_name": ["John Doe", "Jane Smith"],
        }
    )

    try:
        fuzz_merge(
            left,
            right,
            left_on=["first_name", "last_name"],
            right_on="full_name",
        )
    except ValueError as e:
        assert str(e) == "Both left_on and right_on must be list if one of them is list"
