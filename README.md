# txt2sql

[![PyPI version](https://badge.fury.io/py/txt2sql.svg)](https://badge.fury.io/py/txt2sql)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A Python package for helping any tasks related to Text-to-SQL. 
Currently this package provides robust implementations of execution match, intent match, and soft F1 score metrics, specifically designed for comparing SQL query execution results and SQL match for comparing the raw SQL queries.

## Features

- **Execution Match**: Compare execution results with float tolerance handling
- **Intent Match**: Evaluate semantic equivalence of results with flexible matching rules
- **Soft F1 Score**: Calculate continuous similarity scores between result sets
- **SQL Match**: Compare SQL queries with spacing tolerance handling



## Installation

```bash
pip install txt2sql
```

## Quick Start

```python
from txt2sql.metrics import execution_match, intent_match, soft_f1, sql_match

# Example SQL queries
prediction_query = "SELECT id, column_1, column_2 FROM example_data WHERE id IN (1, 2)"
ground_truth_query = "SELECT id, column_1 FROM example_data WHERE id IN (1, 2)"
ground_truth_query_variation = """select id , column_1
from   example_data
where  id in (1 , 2)"""

# Example execution results
prediction = [
    {"id": 1, "column_1": 10.0, "column_2": 15},
    {"id": 2, "column_1": 20.0, "column_2": 15}
]
ground_truth = [
    {"id": 1, "column_1": 10.0},
    {"id": 2, "column_1": 20.0}
]

# Compare queries
sql_match_result = sql_match(prediction_query, ground_truth_query)
sql_match_result_variation = sql_match(ground_truth_query_variation, ground_truth_query)

print(f"SQL Match: {sql_match_result}") # False
print(f"SQL Match variation: {sql_match_result_variation}") # True


# Compare execution results
execution_match_result = execution_match(prediction, ground_truth)
intent_match_result = intent_match(prediction, ground_truth)
soft_f1_result = soft_f1(prediction, ground_truth, ordered=True)

print(f"Execution Match: {execution_match_result}") # False
print(f"Intent Match: {intent_match_result}") # True
print(f"Soft F1 Score: {soft_f1_result}") # 0.8
```

## Metric Definitions

### Execution Match

The execution match metric performs an exact comparison between two sets of SQL execution results, with special handling for floating-point numbers to account for numerical precision issues.

More about the metric can be found on the paper for the BIRD Text-to-SQL benchmark.
https://arxiv.org/pdf/2305.03111

This implementation differs from the benchmark's in that it takes a list of dictionaries not tuples.
In edge where the database result has two columns with the same name,
information will be lost since dictionaries and json object can't hold them both.

```python
from txt2sql.metrics import execution_match

def execution_match(
    prediction: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> bool:
    """
    Compare two lists of dictionaries for equality, handling float values.
    
    Args:
        prediction: List of dictionaries containing the predicted execution results
        ground_truth: List of dictionaries containing the expected execution results
        
    Returns:
        bool: True if the results match, False otherwise
    """
```


### Intent Match

The intent match metric evaluates whether the predicted results satisfy the semantic intent of the ground truth, allowing for various formatting differences and data type variations. Also allows predictions columns to be a superset of the columns in the ground truth.

Derived from intent match description from the paper "NL2SQL is a solved problem... Not!"
https://www.cidrdb.org/cidr2024/papers/p74-floratou.pdf

This implementation of the intent match allows for different formattings for date columns, optionally. Doesn't support hand made rules. Doesn't have side effects on the input data.
```python
from txt2sql.metrics import intent_match

def intent_match(
    prediction: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    normalize_dates: bool = False
) -> bool:
    """
    Check if prediction satisfies the intent in the ground truth.
    
    Args:
        prediction: List of dictionaries containing the predicted execution results
        ground_truth: List of dictionaries containing the expected execution results
        normalize_dates: Whether to normalize different date formats before comparison
        
    Returns:
        bool: True if the results match the intent, False otherwise
    """
```


### Soft F1 Score

The soft F1 score provides a continuous measure of similarity between two result sets, useful for evaluating partial matches and ranking different predictions.

This implementation builds upon previous approaches and introduces several refinements:
Preserves row ordering during duplicate removal to ensure consistent scoring
Uses a deterministic comparison approach that doesn't rely on hash-based sorting
Provides flexibility to evaluate results with or without considering row order


```python
from txt2sql.metrics import soft_f1

def soft_f1(
    prediction: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    ordered: bool = True
) -> float:
    """
    Calculate the F1 score based on sets of predicted results and ground truth results.
    
    Args:
        prediction: List of dictionaries containing the predicted execution results
        ground_truth: List of dictionaries containing the expected execution results
        ordered: Whether to consider row order in the comparison
        
    Returns:
        float: F1 score between 0.0 and 1.0
    """
```

## Development

### Running Tests

The package uses pytest for testing. To run the tests, first install the package with test dependencies:

```bash
pip install -e ".[test]"
```

Then run the tests with coverage report:

```bash
pytest
```

This will run all tests and generate both a terminal coverage report and an HTML coverage report in the `htmlcov` directory.

The test suite includes comprehensive test cases for all metrics, stored as JSON files in `tests/metrics/test_data/`. You can examine these test cases to better understand how each metric behaves in different scenarios.


## References

The implementations in this package build upon approaches from several open source projects:

- **Execution Match**: Based on the [BIRD benchmark execution match evaluation](https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py)
- **Soft F1 Score**: Based on the [BIRD Soft F1 evaluation implementation](https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_f1.py)
- **Intent Match**: Based on the [Archerfish benchmark result set matching](https://github.com/archerfish-bench/benchmark/blob/main/src/benchmark/utils/result_set_match.py)

We appreciate the contributions of these projects to the Text-to-SQL space.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.