from app.crag.models import llm
from app.crag.prompts import generate_prompt
from langchain_core.output_parsers import StrOutputParser
from app.template_parser import ParsedTemplate, TemplateSection
from app.recomposer import CompiledDocument, CompiledSection
from app.retriever.github_fetcher import format_bug_as_context


def process_structured_compliance(template: ParsedTemplate,
    template_fields: list[str], data_records: list[dict]) -> CompiledDocument:
    """
    Compila il template elaborando direttamente una lista di record strutturati.
    """
    print(f"Avvio Compilazione Automatica ({len(data_records)} record)")

    compiled_sections = []

    # Lista dei campi per il Prompt
    fields_list_str = "\n".join([f"- {f}" for f in template_fields])

    # Chain per generator
    chain = generate_prompt | llm | StrOutputParser()

    # Ciclo sui record
    for i, record in enumerate(data_records):
        # Cerca ID: primo campo che contiene "id" nella chiave
        record_id = next(
            (v for k, v in record.items() if "id" in k.lower()),
            f"Record-{i}"
        )
        # Cerca titolo: primo campo che contiene "title" o "name"
        record_title = next(
            (v for k, v in record.items() if any(kw in k.lower() for kw in ("title", "name"))),
            "Untitled"
        )
        # Cerca URL
        source_url = next(
            (v for k, v in record.items() if "url" in k.lower()),
            "unknown"
        )

        print(f"[{i + 1}/{len(data_records)}] Generazione scheda per {record_id}...")

        context_str = format_bug_as_context(record)

        generation = chain.invoke({
            "template_fields": fields_list_str,
            "context": context_str
        })

        # Crea la sezione compilata
        record_section = TemplateSection(
            title = f"Record {record_id} - {record_title}",
            level = 2,
            content = "",
            section_type = "table"
        )

        compiled_sections.append(
            CompiledSection(
                section = record_section,
                generated_content = generation,
                is_abstention = False,
                sources = [source_url]
            )
        )

    return CompiledDocument(template = template, sections = compiled_sections)