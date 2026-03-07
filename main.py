from app.config import BUG_FIXES_TEMPLATE, GITHUB_BUGS_PATH, OUTPUT_DIR
from app.template_parser import parse_template, extract_template_fields
from app.retriever.github_fetcher import load_bugs
from app.compiler.direct_compiler import process_structured_compliance
from app.recomposer import recompose_document, save_document, get_output_path


def main():
    print("TEMPLATE COMPILER")

    # Analisi Template e Campi
    template = parse_template(BUG_FIXES_TEMPLATE)
    template_fields = extract_template_fields(template)

    if not template_fields:
        print("Errore: Nessun campo estratto dal template.")
        return

    # Estrazione Dati
    raw_bugs = load_bugs(GITHUB_BUGS_PATH)[:5]

    if not raw_bugs:
        return

    # Processamento Core (LLM)
    compiled_document = process_structured_compliance(template, template_fields, raw_bugs)

    # Ricomposizione e Salvataggio
    print("\nGenerazione report Markdown finale in corso...")
    final_markdown = recompose_document(compiled_document)

    output_path = get_output_path(template, OUTPUT_DIR)
    save_document(final_markdown, output_path)

    print("\nCompilazione completata con successo!")

if __name__ == "__main__":
    main()