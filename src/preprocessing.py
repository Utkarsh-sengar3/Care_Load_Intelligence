from src.data_validation import parse_numeric_columns

def prepare_observed_records(df, date_column="Date"):
    out = df.loc[~df.isna().all(axis=1)].copy()
    out = parse_numeric_columns(out, date_column=date_column)
    out = out.dropna(subset=[date_column]).sort_values(date_column)
    if out[date_column].duplicated().any():
        raise ValueError("Duplicate dates must be resolved explicitly before modeling.")
    return out.reset_index(drop=True)
