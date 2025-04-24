from typing import Any, Callable, Optional, Literal
from functools import reduce

import pandas as pd
from pandas._typing import AnyArrayLike, IndexLabel, MergeHow
import numpy as np

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
    drop_index: bool = False,
    strategy: Optional[Literal["combine", "aggregate", "hierarchical"]] = None,
    aggregate_func=np.mean,
    **fuzz_kwargs,
):
    """
    Perform a fuzzy merge between two pandas DataFrames or Series based on a similarity score.

    Parameters:
        left (pd.DataFrame | pd.Series): The left DataFrame or Series to merge.
        right (pd.DataFrame | pd.Series): The right DataFrame or Series to merge.
        how (str, optional): Type of merge to be performed. Defaults to 'inner'.
        on (optional): Column or index level name to join on.
            Must be specified if `left_on` and `right_on` are not provided. Defaults to None.
        left_on (optional): Column or index level name in the left DataFrame
            to join on. Defaults to None.
        right_on (optional): Column or index level name in the right DataFrame
            to join on. Defaults to None.
        score_col (str, optional): Name of the column to store the similarity score. Defaults to 'score'.
        scorer (Callable, optional): The similarity scoring function to use. Defaults to 'fuzz.WRatio'.
        score_cutoff (int, optional): The minimum similarity score required to consider a match.
        drop_index (bool, optional): Whether to drop index columns in the result. Defaults to False.
        strategy (str, optional): Strategy for handling multiple columns. Options are
            "combine", "aggregate", or "hierarchical". Defaults to "combine".
        **fuzz_kwargs : Additional support same keyword arguments as `rapidfuzz.process`.
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
    # Value handling
    if how != "inner":
        raise NotImplementedError("Non inner join is supported not supported yet")

    # Handle user parameters set on into left and right
    left_on = left_on if left_on is not None else on
    right_on = right_on if right_on is not None else on
    if left_on is None or right_on is None:
        raise ValueError("Argument 'on' or ('left_on' and 'right_on') are required")

    valid_params = not (isinstance(left_on, list) ^ isinstance(right_on, list))
    is_multiple = isinstance(left_on, list) and isinstance(right_on, list)
    if not valid_params:
        raise ValueError("left_on and right_on must be either both list or or both str")

    # Set default strategy to combine
    strategy = strategy if strategy is not None else "combine"

    if is_multiple and strategy == "hierarchical":
        # Continue to matched df merge logic
        raise NotImplementedError(
            "Multiple columns for this config is not supported yet"
        )

    # Setup Configs
    fuzz_config = _default_config
    fuzz_config.update(
        {
            "scorer": scorer,
            "score_cutoff": score_cutoff,
            **fuzz_kwargs,
        }
    )

    # Coerce columns to string for fuzzy matching
    comp_left = left[left_on].astype(str)
    comp_right = right[right_on].astype(str)

    if is_multiple and strategy == "combine":
        comp_left = comp_left.agg(" ".join, axis=1)
        comp_right = comp_right.agg(" ".join, axis=1)

    if is_multiple and strategy == "aggregate":
        # Must have the same column length
        if len(left_on) != len(right_on):
            raise ValueError(
                "left_on and right_on must have the same number of columns for aggregate strategy"
            )
        agg_matched_df = reduce(
            lambda df, agg: pd.merge(
                df,
                agg,
                on=["left_index", "right_index"],
                how="inner",
            ),
            map(
                lambda col: fuzz_match(
                    left[col[0]].astype(str),
                    right[col[1]].astype(str),
                    score_col="_agg_score",
                    score_cutoff=None,
                    scorer=scorer,
                ),
                zip(left_on, right_on),
            ),
        )
        score_columns = [col for col in agg_matched_df.columns if "_agg_score" in col]
        agg_matched_df[score_col] = aggregate_func(
            agg_matched_df[score_columns], axis=1
        )
        agg_matched_df = agg_matched_df.drop(columns=score_columns)
        # Drop score less than score_cutoff
        if score_cutoff is not None:
            agg_matched_df = agg_matched_df[agg_matched_df[score_col] >= score_cutoff]
        matched_df = agg_matched_df
    else:
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
    if drop_index:
        matched_df = matched_df.drop(columns=["left_index", "right_index"])
    return matched_df
