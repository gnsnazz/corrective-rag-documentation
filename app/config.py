import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROJECT DATA ---
PROJECT_PATH = str(BASE_DIR / "data" / "monai")
DB_DIR = str(BASE_DIR / "data" / "vectorstore" / "project_docs")
OUTPUT_DIR = str(BASE_DIR / "reports")

# --- GITHUB DATA ---
GITHUB_BUGS_PATH = str(BASE_DIR / "data" / "github_bugs.json")

# --- TEMPLATES ---
TEMPLATES_DIR = str(BASE_DIR / "data" / "templates")
BUG_FIXES_TEMPLATE = str(BASE_DIR / "templates" / "techdoc" / "62304" / "bug-fixes-documentation-list.md")

# --- INGESTION ---
ALLOWED_EXTENSIONS = [".py", ".md", ".txt", ".yaml", ".yml", ".cfg", ".toml"]
EXCLUDE_DIRS = ["tests/", ".github/", "__pycache__/", ".git/", "docs/_build/"]

REPO_OWNER = "Project-MONAI"
REPO_NAME = "monai-deploy-app-sdk"

ABSTENTION_MSG = "I am sorry, but the retrieved documents do not contain sufficient information to answer your question."

def format_source(path: str) -> str:
    """Restituisce solo il nome del file dal path completo."""
    return os.path.basename(path) if path else "unknown"

# --- PARAMETRI CRAG ---
MAX_RETRIES = 1
K_CORRECTIVE = 10

STRIP_SIMILARITY_THRESHOLD = 0.50

# --- SOGLIE CRAG ---
CONFIDENCE_UPPER = 0.60
CONFIDENCE_LOWER = 0.30