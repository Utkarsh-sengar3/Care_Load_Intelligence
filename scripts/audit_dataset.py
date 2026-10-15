import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_DATA_PATH, REPORTS_DIR
from src.data_loader import load_raw_data
from src.data_validation import audit_dataframe

def main():
    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DEFAULT_DATA_PATH}")
    df = load_raw_data(DEFAULT_DATA_PATH)
    audit = audit_dataframe(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "data_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(f"Audit report saved to: {output}")

if __name__ == "__main__":
    main()
