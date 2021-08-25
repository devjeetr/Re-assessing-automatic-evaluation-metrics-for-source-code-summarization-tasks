import itertools
import operator
from collections import Counter
from typing import Iterable, Literal, Tuple, Union

import numpy as np
import pandas as pd


def compute_human_ranks(scores: Tuple[float, float], threshold=25) -> Tuple[int, int]:
    """Given a pair of human direct assessment(DA) scores,
       computes the relative ranking. If the difference between
       the two scores is less than the provided threshold,
       the rank is the same.


    Args:
        scores (Tuple[int, int]): A tuple containing the 2 DA scores.
        threshold (int, optional): The threshold of the difference between two scores at which the
                                   the difference is considered (significant). Defaults to 25.

    Returns:
        Tuple[int, int]: The relative ranking of the provided scores
    """
    assert len(scores) == 2
    a, b = scores

    if (a == b) or abs(a - b) < threshold:
        return [1, 1]

    if a > b:
        return [1, 2]

    return [2, 1]


def get_index_pairs(df):
    pairs = list(itertools.combinations(df.index, 2))
    return [pair for pair in pairs if pair[0] != pair[1]]


comparison_variant_c = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "d"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "d"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_d = [
    (operator.lt, operator.lt, "c"),  # <, <
    (operator.lt, operator.eq, "t"),  # <, =
    (operator.lt, operator.gt, "d"),  # <, >
    (operator.eq, operator.lt, "t"),  # =, <
    (operator.eq, operator.eq, "c"),
    (operator.eq, operator.gt, "t"),  # =, >
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]

comparison_variant_b = [
    (operator.lt, operator.lt, "c"),
    (operator.lt, operator.eq, "t"),
    (operator.lt, operator.gt, "d"),
    (operator.gt, operator.lt, "d"),
    (operator.gt, operator.eq, "t"),
    (operator.gt, operator.gt, "c"),
]


def compute_rank_pair_type(
    human_ranking: Iterable[Union[int, float]],
    metric_ranking: Iterable[Union[int, float]],
) -> Union[Literal["c"], Literal["d"], None]:
    comparison_table = comparison_variant_b

    for h_op, m_op, outcome in comparison_table:
        if h_op(*human_ranking) and m_op(*metric_ranking):
            return outcome

    return "-"


def kendalls_tau_darr(
    df: pd.DataFrame,
    groupby=["question_id", "user_id"],
    human_col="ranking",
    metric_col="score",
    threshold=25,
):
    """Computes the Kendall's Tau formulation for da-RR, as presented
    by Ma et al. (2019), "Results of the WMT19 metrics shared task:
    Segment-level and strong MT systems pose big challenges."

     It is given by:

                |Concordant| - |Discordant|
         \tau = -------------------------------
                |Concordant| + |Discordant|


     where:
         ╔═══════╦═════════╦═════════╦═════════╦═════════╗
         ║       ║         ║ metric  ║         ║         ║
         ╠═══════╬═════════╬═════════╬═════════╬═════════╣
         ║       ║         ║ s1 < s2 ║ s1 = s2 ║ s1 > s2 ║
         ╠═══════╬═════════╬═════════╬═════════╬═════════╣
         ║ human ║ s1 < s2 ║ Conc    ║ Disc    ║ Disc    ║
         ║       ╠═════════╬═════════╬═════════╬═════════╣
         ║       ║ s1 = s2 ║ -       ║ -       ║ -       ║
         ║       ╠═════════╬═════════╬═════════╬═════════╣
         ║       ║ s1 > s2 ║ Disc    ║ Disc    ║ Conc    ║
         ╚═══════╩═════════╩═════════╩═════════╩═════════╝

     args:
         human_rankings - 2d numpy array of human rankings
         metric_rankings - 2d numpy array of metric rankings, corresponding
                           to the human rankings
    """
    counts = Counter()
    grouped = df.groupby(groupby)
    for group in grouped.groups:
        current_group = grouped.get_group(group)
        pairs = get_index_pairs(current_group)
        pair_types = []
        for pair in pairs:
            pair_df = current_group[current_group.index.isin(pair)]
            human_scores = pair_df[human_col]
            metric_scores = pair_df[metric_col]

            human_ranks = compute_human_ranks(human_scores, threshold=threshold)
            metric_ranks = metric_scores.rank(method="min")

            pair_type = compute_rank_pair_type(human_ranks, metric_ranks)
            pair_types.append(pair_type)

        counts.update(pair_types)
    concordant_pairs = counts["c"]
    discordant_pairs = counts["d"]
    ties = counts["t"]
    tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs + ties
    )
    return {
        "tau": tau,
        "concordant": concordant_pairs,
        "discordant": discordant_pairs,
        "ties": ties,
    }
