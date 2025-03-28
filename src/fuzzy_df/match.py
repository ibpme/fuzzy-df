from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from rapidfuzz import process

from fuzzy_df.config import _default_config


def fuzz_match(
    comp_left: pd.Series,
    comp_right: pd.Series,
    score_col="score",
    scorer: Callable = _default_config["scorer"],
    score_cutoff: Optional[Any] = _default_config["score_cutoff"],
    **fuzz_kwargs,
) -> pd.DataFrame:
    """
    Perform fuzzy matching between two pandas Series and return a DataFrame of matches.

    This function computes similarity scores between elements of `comp_left` and `comp_right`
    using a fuzzy matching algorithm. It returns a DataFrame containing the indices of matched
    elements from both Series along with their corresponding similarity scores.

    Parameters:
        comp_left (pd.Series): The left-hand Series to compare.
        comp_right (pd.Series): The right-hand Series to compare.
        score_col (str, optional): The name of the column in the output DataFrame that will
            store the similarity scores. Defaults to 'score'.
        scorer (Callable, optional): The similarity scoring function to use. Defaults to 'fuzz.WRatio'.
        score_cutoff (int, optional): The minimum similarity score required to consider a match.
        **fuzz_kwargs : Additional support same keyword arguments as `rapidfuzz.process`:
            https://rapidfuzz.github.io/RapidFuzz/Usage/process.html#rapidfuzz.process

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'left_index': The index of the matched element in `comp_left`.
            - 'right_index': The index of the matched element in `comp_right`.
            - `<score_col>`: The similarity score of the match (column name is determined by `score_col`).

    Example:
        >>> comp_left = pd.Series(["apple", "banana", "cherry"])
        >>> comp_right = pd.Series(["apples", "bananas", "grape"])
        >>> fuzz_match(comp_left, comp_right, score_cutoff=70)
           left_index  right_index  score
        0           0            0   90.0
        1           1            1   85.0
    """

    fuzz_config = _default_config
    fuzz_config.update(
        {
            "scorer": scorer,
            "score_cutoff": score_cutoff,
            **fuzz_kwargs,
        }
    )
    scores = process.cdist(
        comp_left,
        comp_right,
        **fuzz_config,
    )

    match_indices = np.nonzero(scores)
    matched_df = pd.DataFrame(
        np.array((*match_indices, scores[match_indices])).T,
        columns=["left_index", "right_index", score_col],
    ).astype({"left_index": int, "right_index": int, score_col: float})
    return matched_df
