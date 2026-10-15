import pandas as pd

def audit_dataframe(df: pd.DataFrame, date_column: str = "Date") -> dict:
    # Conservative audit: never imputes values or creates missing dates.
    result = {
        "source_rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "completely_empty_rows": int(df.isna().all(axis=1).sum()),
        "missing_values_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_dates": 0,
        "invalid_date_rows": None,
        "first_date": None,
        "last_date": None,
        "observed_unique_dates": None,
        "calendar_days_inclusive": None,
        "missing_calendar_dates": None,
        "date_gap_days_summary": None,
    }
    if date_column not in df.columns:
        result["date_error"] = f"Date column '{date_column}' not found"
        return result
    dates = pd.to_datetime(df[date_column], errors="coerce")
    populated = ~df.isna().all(axis=1)
    valid_dates = dates[populated].dropna()
    result["invalid_date_rows"] = int(dates[populated].isna().sum())
    if valid_dates.empty:
        return result
    unique_dates = pd.DatetimeIndex(valid_dates.drop_duplicates().sort_values())
    result["duplicate_dates"] = int(valid_dates.duplicated().sum())
    result["first_date"] = str(unique_dates.min().date())
    result["last_date"] = str(unique_dates.max().date())
    result["observed_unique_dates"] = int(len(unique_dates))
    calendar_days = int((unique_dates.max() - unique_dates.min()).days + 1)
    result["calendar_days_inclusive"] = calendar_days
    result["missing_calendar_dates"] = int(calendar_days - len(unique_dates))
    if len(unique_dates) > 1:
        gaps = unique_dates.to_series().diff().dropna().dt.days
        result["date_gap_days_summary"] = {
            "minimum_gap_days": int(gaps.min()),
            "median_gap_days": float(gaps.median()),
            "maximum_gap_days": int(gaps.max()),
            "gap_counts": {str(int(k)): int(v) for k, v in gaps.value_counts().sort_index().items()},
        }
    return result

def parse_numeric_columns(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == date_column:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif out[col].dtype == object:
            cleaned = out[col].astype("string").str.replace(",", "", regex=False).str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            source_has_value = out[col].notna() & out[col].astype(str).str.strip().ne("")
            failures = source_has_value & converted.isna()
            if failures.any():
                examples = out.loc[failures, col].astype(str).head(5).tolist()
                raise ValueError(f"Could not convert '{col}' to numeric. Examples: {examples}")
            out[col] = converted
    return out
