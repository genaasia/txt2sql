from datetime import datetime
from typing import Any, Dict, List


def remove_duplicates(list_of_dicts: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for d in list_of_dicts:
        d_str = str(d)
        if d_str not in seen:
            seen.add(d_str)
            result.append(d)
    return result


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


def remove_quotes(string: Any) -> str:
    """Function to remove leading and trailing quotes"""
    if isinstance(string, str):
        return string.strip('"')
    return string


def normalize_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all keys in prediction and ground_truth to lowercase for case-insensitive comparison
    Remove quotes in string values in prediction and ground_truth
    """
    return {(k.lower() if k is not None else None): remove_quotes(v) for k, v in row.items()}
