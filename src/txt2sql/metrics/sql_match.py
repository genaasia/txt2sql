import io
from contextlib import redirect_stdout

import sqlfluff


def normalize_query(query: str, dialect: str) -> str:
    query = " ".join(query.split())
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            formatted_query = sqlfluff.fix(query, dialect=dialect)
        return formatted_query
    except:
        return query


def sql_match(pred_sql: str, label_sql: str, dialect: str = "sqlite") -> bool:
    return normalize_query(pred_sql, dialect) == normalize_query(label_sql, dialect)
