import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

REPO_PATH = str(BASE_DIR / "data" / "transformers" / "docs" / "source" / "en")
DB_DIR = str(BASE_DIR / "data" / "vectorstore" / "transformers_md")

ABSTENTION_MSG = "I am sorry, but the retrieved documents do not contain sufficient information to answer your question."

def format_source(path: str) -> str:
    """Restituisce solo il nome del file dal path completo."""
    return os.path.basename(path) if path else "unknown"

# Parametri CRAG
CONFIDENCE_THRESHOLD = 0.5  # Se < 50% dei docs sono validi -> Corrective Search
MAX_RETRIES = 1             # Numero di cicli correttivi
K_CORRECTIVE = 10           # Documenti giro correttivo

STRIP_SIMILARITY_THRESHOLD = 0.55  # Soglia cosine similarity per tenere uno strip