from typing import Any, Callable, Optional

import pandas as pd
from pandas._typing import AnyArrayLike, IndexLabel, MergeHow

from fuzzy_df.config import _default_config
from fuzzy_df.match import fuzz_match


def fuzz_merge(
    left: pd.DataFrame | pd.Series,
    right: pd.DataFrame | pd.Series,
    how: MergeHow = "inner",
    on: IndexLabel | AnyArrayLike | None = None,
    left_on: IndexLabel | AnyArrayLike | None = None,
    right_on: IndexLabel | AnyArrayLike | None = None,
    score_col="score",
    scorer: Callable = _default_config["scorer"],
    score_cutoff: Optional[Any] = _default_config["score_cutoff"],
    **fuzz_kwargs,
):
    """
    Perform a fuzzy merge between two pandas DataFrames or Series based on a similarity score.

    Parameters:
        left (pd.DataFrame | pd.Series): The left DataFrame or Series to merge.
        right (pd.DataFrame | pd.Series): The right DataFrame or Series to merge.
        how (MergeHow, optional): Type of merge to be performed. Defaults to 'inner'.
        on (IndexLabel | AnyArrayLike | None, optional): Column or index level name to join on.
            Must be specified if `left_on` and `right_on` are not provided. Defaults to None.
        left_on (IndexLabel | AnyArrayLike | None, optional): Column or index level name in the left DataFrame
            to join on. Defaults to None.
        right_on (IndexLabel | AnyArrayLike | None, optional): Column or index level name in the right DataFrame
            to join on. Defaults to None.
        score_col (str, optional): Name of the column to store the similarity score. Defaults to 'score'.
        scorer (Callable, optional): The similarity scoring function to use. Defaults to 'fuzz.WRatio'.
        score_cutoff (int, optional): The minimum similarity score required to consider a match.
        **fuzz_kwargs : Additional support same keyword arguments as `rapidfuzz.process`:
            https://rapidfuzz.github.io/RapidFuzz/Usage/process.html#rapidfuzz.process

    Returns:
        pd.DataFrame: A DataFrame containing the merged results with fuzzy matching applied.

    Raises:
        ValueError: If `on`, `left_on`, or `right_on` are not provided.
        NotImplementedError: If multiple columns are specified for `left_on` or `right_on`.

    Example:
        >>> left = pd.DataFrame(
            {"id_left": [1, 2], "name_left": ["foo", "bar"]})
        >>> right = pd.DataFrame(
            {"id_right": [3, 4, 5, 6], "name_right": ["baz", "bear", "fool", "food"]})
        >>> merged = fuzz_merge(left, right, left_on="name_left",
                    right_on="name_right", score_cutoff=70)
    ```
        id_right name_right  id_left name_left  left_index  right_index      score
    2         4       bear        2       bar           1            1  85.714287
    0         5       fool        1       foo           0            2  85.714287
    1         6       food        1       foo           0            3  85.714287
    ```

    Notes:
        - The function uses fuzzy matching to compare the specified columns in the left and right DataFrames.
        - The `fuzz_match` function is expected to handle the actual fuzzy matching logic.
        - The resulting DataFrame includes the similarity score and merges the left and right DataFrames based on
          the specified `how` parameter.
    """
    if how != "inner":
        raise NotImplementedError("Non inner join is supported not supported yet")
    left_on = left_on if left_on is not None else on
    right_on = right_on if right_on is not None else on
    if left_on is None or right_on is None:
        raise ValueError("on, left_on and right_on are required")
    if isinstance(left_on, list):
        raise NotImplementedError("Multiple columns not supported yet")
    if isinstance(right_on, list):
        raise NotImplementedError("Multiple columns not supported yet")

    comp_left = left[left_on].astype(str)
    comp_right = right[right_on].astype(str)

    fuzz_config = _default_config
    fuzz_config.update(
        {
            "scorer": scorer,
            "score_cutoff": score_cutoff,
            **fuzz_kwargs,
        }
    )

    matched_df = fuzz_match(comp_left, comp_right, score_col, **fuzz_config)

    matched_df = pd.merge(
        left.reset_index(drop=True),
        matched_df,
        left_index=True,
        right_on="left_index",
        how=how,
    )
    matched_df = pd.merge(
        right.reset_index(drop=True),
        matched_df,
        left_index=True,
        right_on="right_index",
        how=how,
    )
    return matched_df
