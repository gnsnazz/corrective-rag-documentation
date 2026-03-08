import os
import requests
import json
from pathlib import Path
from app.config import GITHUB_BUGS_PATH, REPO_OWNER, REPO_NAME
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"

def get_headers() -> dict:
    """
    Genera gli header, includendo l'autenticazione se disponibile.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def fetch_issue_comments(owner: str, repo: str, issue_number: int) -> list[dict]:
    """
    Scarica i commenti di una issue.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}/comments"

    response = requests.get(url, headers = get_headers())

    if response.status_code != 200:
        return []

    comments_raw = response.json()

    comments = []
    for c in comments_raw:
        comments.append({
            "author": c.get("user", {}).get("login", "unknown"),
            "body": c.get("body", ""),
            "created_at": c.get("created_at")
        })

    return comments


def fetch_bugs(owner: str, repo: str, state: str = "all", label: str = "bug", per_page: int = 100) -> list[dict]:
    """
    Scarica tutte le issues con label 'bug' da un repository GitHub.
    """
    print(f"Ricerca bug ({state}) su {owner}/{repo}...")

    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues"
    bugs = []
    page = 1

    while True:
        params = {
            "state": state,
            "labels": label,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc"
        }

        response = requests.get(url, params = params, headers = get_headers())

        if response.status_code != 200:
            print(f"Errore API: {response.status_code} - {response.text}")
            break

        raw_issues = response.json()

        if not raw_issues:
            break

        for issue in raw_issues:

            # GitHub restituisce anche le PR → filtra
            if "pull_request" in issue:
                continue

            issue_number = issue["number"]

            comments = fetch_issue_comments(owner, repo, issue_number)

            bug = {
                "bug_id": f"#{issue_number}",
                "title": issue["title"],
                "description": issue["body"] if issue["body"] else "No description provided.",
                "state": issue["state"],
                "created_at": issue["created_at"],
                "closed_at": issue.get("closed_at"),
                "labels": [l["name"] for l in issue.get("labels", [])],
                "url": issue["html_url"],
                "author": issue.get("user", {}).get("login", "unknown"),
                "comments": comments
            }

            bugs.append(bug)

        print(f"Pagina {page} scaricata ({len(raw_issues)} issues)")
        page += 1

    print(f"\nTotale bug trovati: {len(bugs)}")
    return bugs


def save_bugs(bugs: list[dict], output_path: str) -> None:
    """Salva la lista di bug in un file JSON."""
    Path(output_path).parent.mkdir(parents = True, exist_ok = True)

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(bugs, f, indent = 4, ensure_ascii = False)

    print(f"Salvati in: {output_path}")


def load_bugs(json_path: str) -> list[dict]:
    """
    Carica i bug dal JSON salvato dal fetcher.
    Usato dal main per leggere i dati scaricati.
    """
    if not os.path.exists(json_path):
        print(f"File bug non trovato: {json_path}")
        print("Esegui prima: python github_fetcher.py")
        return []

    with open(json_path, "r", encoding = "utf-8") as f:
        bugs = json.load(f)

    print(f"Caricati {len(bugs)} bug da {os.path.relpath(json_path)}")
    return bugs

def format_bug_as_context(bug: dict) -> str:
    """
    Formatta il dizionario del bug in una stringa di contesto per l'LLM.
    """
    return json.dumps(bug, indent = 2)

if __name__ == "__main__":

    bugs = fetch_bugs(REPO_OWNER, REPO_NAME, state = "all")
    save_bugs(bugs, GITHUB_BUGS_PATH)