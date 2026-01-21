from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

REPO_PATH = str(BASE_DIR / "data" / "transformers" / "docs" / "source" / "en")
DB_DIR = str(BASE_DIR / "data" / "vectorstore" / "transformers_md")

ABSTENTION_MSG = "I am sorry, but the retrieved documents do not contain sufficient information to answer your question."