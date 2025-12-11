import os
import re

def sanitize_filename(topic: str) -> str:
    """Pulisce la stringa per creare un nome file valido."""
    filename = re.sub(r'[^\w\s-]', '', topic)
    filename = re.sub(r'[-\s]+', '_', filename).strip().lower()
    return f"{filename}.md"

def save_documentation(content: str, topic: str, output_folder: str = "output_docs") -> str:
    """
    Salva il contenuto in un file Markdown nella cartella specificata.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = sanitize_filename(topic)
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        # Intestazione standard
        f.write(f"# {topic}\n")
        f.write(f"**Automated Documentation via CRAG**\n")
        f.write("---\n\n")
        f.write(content)

    return os.path.abspath(filepath)