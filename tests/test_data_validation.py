import pandas as pd
from src.data_validation import audit_dataframe, parse_numeric_columns

def test_audit_counts_empty_rows_and_calendar_gaps():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-03", None], "Count": ["1,000", "1,100", None]})
    result = audit_dataframe(df)
    assert result["source_rows"] == 3
    assert result["completely_empty_rows"] == 1
    assert result["missing_calendar_dates"] == 1

def test_numeric_conversion_removes_commas():
    df = pd.DataFrame({"Date": ["2024-01-01"], "Count": ["1,234"]})
    result = parse_numeric_columns(df)
    assert result.loc[0, "Count"] == 1234
