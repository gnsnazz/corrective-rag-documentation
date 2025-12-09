import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DB_DIR = "data/vectorstore/sklearn_md"
REPO_PATH = "data/sklearn_repo"