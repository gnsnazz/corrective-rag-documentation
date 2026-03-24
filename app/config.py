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
BUG_FIXES_TEMPLATE = str(BASE_DIR / "templates" / "techdoc" / "62304" / "bug-fixes-documentation-list.md")
REQUIREMENTS_TEMPLATE = str(BASE_DIR / "templates" / "techdoc" / "62304" / "software-requirements-list.md")
SOFTWARE_LIST_TEMPLATE = str(BASE_DIR / "templates" / "qms" / "software_validation" / "software-list.md")

TEMPLATES = {
    "requirements": {
        "path": REQUIREMENTS_TEMPLATE,
        "query_suffix": "software requirements features capabilities dependencies configuration specifications"
    },
    "software-list": {
        "path": SOFTWARE_LIST_TEMPLATE,
        "query_suffix": "software components versions dependencies validation manufacturer"
    }
}

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
K_BASE = 10

STRIP_SIMILARITY_THRESHOLD = 0.45
