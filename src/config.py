from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_DATA_PATH = DATA_RAW_DIR / "HHS_Unaccompanied_Alien_Children_Program.csv"
DATE_COLUMN = "Date"
TARGET_COLUMNS = ["Children in HHS Care", "Children discharged from HHS Care"]
