import io
from contextlib import redirect_stdout

import sqlfluff
import sqlparse


def normalize_query(query: str, dialect: str) -> str:
    """Normalize spacing, casing, identation etc. and remove comments from a SQL query"""
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            formatted_query = sqlfluff.fix(query, dialect=dialect)
        formatted_query = sqlparse.format(
            formatted_query,
            strip_comments=True,
            keyword_case="upper",
            reindent=True,
        )
        return formatted_query
    except:
        return query


def sql_match(pred_sql: str, label_sql: str, dialect: str = "sqlite") -> bool:
    """Normalize two queries and check if they are the same query"""
    return normalize_query(pred_sql, dialect) == normalize_query(label_sql, dialect)
