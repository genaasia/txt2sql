"""Intent match metric for evaluating Text-to-SQL models,
based on the execution results as list of Python dictionaries.

Derived from intent match description from the paper NL2SQL is a solved problem... Not!
https://www.cidrdb.org/cidr2024/papers/p74-floratou.pdf

This version:
Allows for different formattings for date columns, optionally.
Doesn't support rules. 
Doesn't have side effects.
"""

import decimal
from datetime import datetime
from typing import Any, Dict, List

from .utils import remove_duplicates


def parse_date_string(date_string: str) -> str:
    """Parse the string if it is in a date time format
    Return as it is if it isn't
    """
    formats = [
        "%Y-%m",  # for "2023-10"
        "%Y-%m-%d",  # for "2023-10-01"
        "%Y-%m-%dT%H:%M:%S",  # for "2023-10-01T00:00:00"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    return date_string


def remove_quotes(string: str) -> str:
    """Function to remove leading and trailing quotes"""
    if isinstance(string, str):
        return string.strip('"')
    return string


def normalize_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in prediction and ground_truth to lowercase for case-insensitive comparison
    Remove quotes in string values in prediction and ground_truth
    """
    return {
        (k.lower() if k is not None else None): remove_quotes(v) for k, v in row.items()
    }


def intent_match(
    prediction: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    normalize_dates: bool = False,
) -> bool:
    """Check if prediction satisfies the intent in the ground truth.
    Row counts should be same but order doesn't matter.
    Key names or order doesn't matter.
    Different types of the same values are allowed
    """
    ground_truth = [normalize_dict(row) for row in ground_truth]
    prediction = [normalize_dict(row) for row in prediction]

    if len(ground_truth) > 0 and len(prediction) > 0:
        ground_truth_cols = ground_truth[0].keys()
        prediction_cols = prediction[0].keys()

        # Proceed with de-dup only when prediction and ground_truth cols are same
        if ground_truth_cols == prediction_cols and len(ground_truth_cols) == 1:
            ground_truth = remove_duplicates(ground_truth)
            prediction = remove_duplicates(prediction)

    if len(ground_truth) != len(prediction):
        return False

    formatting_memory = {}

    def is_match(predicted_val, ground_truth_val):
        """This function compares expected_val and actual_val with relevant datatype conversion"""
        if normalize_dates:
            if isinstance(predicted_val, str):
                predicted_val = formatting_memory.setdefault(
                    predicted_val, parse_date_string(predicted_val)
                )
            if isinstance(ground_truth_val, str):
                ground_truth_val = formatting_memory.setdefault(
                    predicted_val, parse_date_string(ground_truth_val)
                )

        if (predicted_val is None or predicted_val == 0) and (
            ground_truth_val is None or ground_truth_val == 0
        ):
            return True
        if isinstance(predicted_val, (int, float, decimal.Decimal)) and isinstance(
            ground_truth_val, (int, float, decimal.Decimal)
        ):
            return ground_truth_val is not None and (
                str(round(predicted_val, 1)) == str(round(ground_truth_val, 1))
            )
        return str(predicted_val) == str(ground_truth_val)

    def match_all_cols(expected_row, actual_row):
        """Run is_match for all columns in  the expected row against all rows in actual row"""
        for column in expected_row.keys():
            if column in actual_row:
                if not is_match(expected_row[column], actual_row[column]):
                    return False
            else:
                column_matched = any(
                    is_match(expected_row[column], actual_val)
                    for actual_val in actual_row.values()
                )
                if not column_matched:
                    return False
        return True

    skip_indices = set()
    for ground_truth_idx, ground_truth_row in enumerate(ground_truth):
        if ground_truth_idx not in skip_indices and match_all_cols(
            ground_truth_row, prediction[ground_truth_idx]
        ):
            skip_indices.add(ground_truth_idx)
        else:
            for prediction_idx, prediction_row in enumerate(prediction):
                if prediction_idx in skip_indices:
                    continue
                if match_all_cols(ground_truth_row, prediction_row):
                    skip_indices.add(prediction_idx)
                    break
            else:
                return False
    return True
