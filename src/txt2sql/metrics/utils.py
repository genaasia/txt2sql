from typing import Dict, List


def remove_duplicates(list_of_dicts: List[Dict]) -> List[Dict]:
    seen = set()
    result = []
    for d in list_of_dicts:
        d_str = str(d)
        if d_str not in seen:
            seen.add(d_str)
            result.append(d)
    return result
