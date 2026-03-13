import sys
from app.config import BUG_FIXES_TEMPLATE, REQUIREMENTS_TEMPLATE, GITHUB_BUGS_PATH, OUTPUT_DIR, REPO_NAME
from app.template_parser import parse_template, extract_template_fields
from app.retriever.github_fetcher import load_bugs
from app.compiler.direct_compiler import process_structured_compliance
from app.compiler.crag_compiler import process_crag_compliance
from app.recomposer import recompose_document, save_document, get_output_path


def run_bug_fixes():
    print("Bug Fixes (Dati Strutturati)")

    template = parse_template(BUG_FIXES_TEMPLATE)
    template_fields = extract_template_fields(template)

    if not template_fields:
        print("Errore: Nessun campo estratto dal template.")
        return

    raw_bugs = load_bugs(GITHUB_BUGS_PATH)[:7]
    if not raw_bugs:
        return

    compiled_document = process_structured_compliance(template, template_fields, raw_bugs)

    print("\nGenerazione report Markdown finale in corso...")
    final_markdown = recompose_document(compiled_document, transposed = True)

    output_path = get_output_path(template, OUTPUT_DIR)
    save_document(final_markdown, output_path)
    print("\nCompilazione completata con successo!")


def run_requirements():
    print("Software Requirements (CRAG)")

    template = parse_template(REQUIREMENTS_TEMPLATE)
    template_fields = extract_template_fields(template)

    if not template_fields:
        print("Errore: Nessun campo estratto dal template.")
        return

    compiled_document = process_crag_compliance(template, template_fields, repo_name = REPO_NAME)

    print("\nGenerazione report Markdown finale in corso...")
    final_markdown = recompose_document(compiled_document)

    output_path = get_output_path(template, OUTPUT_DIR)
    save_document(final_markdown, output_path)
    print("\nCompilazione completata con successo!")


def main():
    print("TEMPLATE COMPILER")

    if len(sys.argv) < 2:
        print("Seleziona il caso:")
        print("  1 - Bug Fixes")
        print("  2 - Software Requirements")
        command = input("Scelta: ").strip()
        command = {"1": "bugs", "2": "requirements"}.get(command, command)
    else:
        command = sys.argv[1]

    if command == "bugs":
        run_bug_fixes()
    elif command == "requirements":
        run_requirements()
    else:
        print(f"Comando non valido: '{command}'")


if __name__ == "__main__":
    main()