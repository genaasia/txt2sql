"""Soft F1 Score metric for evaluating Text-to-SQL models,
based on the execution results as list of Python dictionaries.

Gives a continous value for similarity of two execution results,
based on the definition here: https://github.com/bird-bench/mini_dev

Original implementation changes the ordering of rows that have partial matching,
so results in wrong scores.
Also it depends on sorting based on hash values which changes every run,
so it is non-deterministic score.

This version:
Removes duplicates without changing the ordering,
so it doesn't have the problems in the original implementation,
also supports checking for ordered and unordered comparison.
"""

from typing import Any, Dict, List, Tuple

from .utils import remove_duplicates


def calculate_row_match(
    predicted_row: Dict[str, Any], ground_truth_row: Dict[str, Any]
) -> Tuple[float, float, float]:
    """Calculate the matching percentage for a single row."""
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row.values():
        if pred_val in ground_truth_row.values():
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row.values():
        if truth_val not in predicted_row.values():
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage


def soft_f1(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    ordered: bool = True,
) -> float:
    """Calculate the F1 score based on sets of predicted results and ground truth results,
    where each element (dict) represents a row from the database with multiple columns.
    """
    # if both predicted and ground_truth are empty, return 1.0 for f1_score
    if not predicted and not ground_truth:
        return 1.0

    # Drop duplicates
    predicted_set = remove_duplicates(predicted)
    ground_truth_set = remove_duplicates(ground_truth)

    # convert back to list
    predicted = list(predicted_set)
    ground_truth = list(ground_truth_set)

    # Calculate matching scores for each possible pair
    match_scores = []
    pred_only_scores = []
    truth_only_scores = []

    for i, gt_row in enumerate(ground_truth):
        # rows only in the ground truth results
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        if ordered:
            pred_row = predicted[i]
            match_score, pred_only_score, truth_only_score = calculate_row_match(
                pred_row, gt_row
            )
        else:
            max_match_score = 0
            for pred_row in predicted:
                match_score, pred_only_score, truth_only_score = calculate_row_match(
                    pred_row, gt_row
                )
                if match_score == 1:
                    max_match_score = match_score
                    break
                max_match_score = max(max_match_score, match_score)
            match_score = max_match_score
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    # rows only in the predicted results
    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return f1_score
