from app.crag.models import llm
from app.crag.prompts import bug_generate_prompt
from langchain_core.output_parsers import StrOutputParser
from app.template_parser import ParsedTemplate, TemplateSection
from app.recomposer import CompiledDocument, CompiledSection
from app.retriever.github_fetcher import format_bug_as_context, RECORD_ID_KEYS, RECORD_TITLE_KEYS, RECORD_URL_KEYS


def extract_record_meta(record: dict, index: int) -> tuple[str, str, str]:
    """
    Estrae ID, titolo e URL da un record generico.

    Cerca le chiavi in ordine di priorità (RECORD_*_KEYS).
    """
    record_id = next(
        (record[k] for k in RECORD_ID_KEYS if k in record),
        f"Record-{index}"
    )
    record_title = next(
        (record[k] for k in RECORD_TITLE_KEYS if k in record),
        "Untitled"
    )
    source_url = next(
        (record[k] for k in RECORD_URL_KEYS if k in record),
        "unknown"
    )
    return record_id, record_title, source_url


def process_structured_compliance(template: ParsedTemplate, template_fields: list[str], data_records: list[dict]) -> CompiledDocument:
    """
    Compila il template elaborando direttamente una lista di record strutturati.

    Per ogni record:
    - formatta il contesto con format_bug_as_context()
    - invoca l'LLM per generare una tabella | Field | Value |

    Il documento restituito viene poi passato a recompose_document(..., transposed=True)
    per produrre la tabella finale con bug sulle righe e campi sulle colonne.
    """
    print(f"Avvio Compilazione Automatica ({len(data_records)} record)")

    compiled_sections = []
    fields_list_str = "\n".join(f"- {f}" for f in template_fields)
    chain = bug_generate_prompt | llm | StrOutputParser()

    for i, record in enumerate(data_records):
        record_id, record_title, source_url = extract_record_meta(record, i)
        print(f"[{i + 1}/{len(data_records)}] Generazione scheda per {record_id}...")

        # Sezione fittizia: rappresenta un singolo record nel documento finale
        record_section = TemplateSection(
            title = f"{record_id} - {record_title}",
            level = 2,
            content = "",
            section_type = "table"
        )

        context_str = format_bug_as_context(record)

        try:
            generation = chain.invoke({
                "template_fields": fields_list_str,
                "context": context_str
            })
            compiled_sections.append(CompiledSection(
                section = record_section,
                generated_content = generation,
                is_abstention = False,
                sources = [source_url]
            ))

        except Exception as e:
            print(f"  Errore su {record_id}: {e}")
            compiled_sections.append(CompiledSection(
                section = record_section,
                generated_content = "",
                is_abstention = True,
                sources = [source_url]
            ))

    return CompiledDocument(template = template, sections = compiled_sections)