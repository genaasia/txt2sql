import pytest

from txt2sql.metrics import execution_match, intent_match, soft_f1, sql_match


def load_test_cases(file_name):
    import json
    from pathlib import Path

    test_data_path = Path(__file__).parent / "test_data" / file_name
    with open(test_data_path) as f:
        cases = json.load(f)

    all_cases = []
    for category in cases.values():
        all_cases.extend(category)
    return all_cases


@pytest.mark.parametrize("case", load_test_cases("execution_match_cases.json"))
def test_execution_match(case):
    result = execution_match(case["prediction"], case["ground_truth"])
    assert result == case["expected"], f"Failed on case: {case['name']}"


@pytest.mark.parametrize("case", load_test_cases("intent_match_cases.json"))
def test_intent_match(case):
    result = intent_match(
        case["prediction"],
        case["ground_truth"],
        normalize_dates=case.get("normalize_dates", False),
        use_dynamic_precision=case.get("use_dynamic_precision", False),
    )
    assert result == case["expected"], f"Failed on case: {case['name']}"


@pytest.mark.parametrize("case", load_test_cases("soft_f1_cases.json"))
def test_soft_f1(case):
    result = soft_f1(case["prediction"], case["ground_truth"], case["ordered"])
    assert result == pytest.approx(
        case["expected"], rel=0.01
    ), f"Failed on case: {case['name']}"


@pytest.mark.parametrize("case", load_test_cases("sql_match_cases.json"))
def test_sql_match(case):
    result = sql_match(case["prediction"], case["ground_truth"])
    assert result == case["expected"], f"Failed on case: {case['name']}"
