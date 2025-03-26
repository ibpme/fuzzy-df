import numpy as np
import pandas as pd
from rapidfuzz import process


def fuzz_match(comp_left: pd.Series, comp_right: pd.Series, score_col='score', score_cutoff=80):
    """
    Perform fuzzy matching between two pandas Series and return a DataFrame of matches.

    This function computes similarity scores between elements of `comp_left` and `comp_right`
    using a fuzzy matching algorithm. It returns a DataFrame containing the indices of matched
    elements from both Series along with their corresponding similarity scores.

    Args:
        comp_left (pd.Series): The left-hand Series to compare.
        comp_right (pd.Series): The right-hand Series to compare.
        score_col (str, optional): The name of the column in the output DataFrame that will
            store the similarity scores. Defaults to 'score'.
        score_cutoff (int, optional): The minimum similarity score required to consider a match.
            Defaults to 80.

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
    scores = process.cdist(comp_left, comp_right, score_cutoff=score_cutoff)
    match_indices = np.nonzero(scores)
    matched_df = pd.DataFrame(
        np.array((*match_indices, scores[match_indices])).T,
        columns=['left_index', 'right_index', score_col],
    ).astype({'left_index': int, 'right_index': int, score_col: float})
    return matched_df
