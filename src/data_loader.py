import pandas as pd
from src.config import DEFAULT_DATA_PATH

def load_raw_data(path=DEFAULT_DATA_PATH) -> pd.DataFrame:
    # Load source as-is, including blank lines, so the audit can report them.
    return pd.read_csv(str(path), skip_blank_lines=False)
