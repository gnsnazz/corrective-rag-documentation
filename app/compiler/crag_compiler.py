from app.crag.graph import build_crag_graph
from app.template_parser import ParsedTemplate, TemplateSection
from app.recomposer import CompiledDocument, CompiledSection
from app.config import ABSTENTION_MSG, REPO_NAME
from pathlib import Path


def build_requirements_query(template_title: str) -> str:
    """
    Costruisce la macro-query per recuperare tutti i requisiti dal vector store.
    """
    return f"{REPO_NAME} {template_title} software requirements features capabilities dependencies configuration specifications"


def process_crag_compliance(template: ParsedTemplate, template_fields: list[str]) -> CompiledDocument:
    """
    Compila il template requirements con approccio single-pass:
    1. Una sola macro-query al CRAG per recuperare tutti i chunk rilevanti
    2. Il grader filtra i documenti rilevanti
    3. Il nodo generate produce l'intera tabella dei requisiti
    """
    print(f"Avvio Compilazione CRAG Single-Pass - {template.title}")

    app = build_crag_graph()

    query = build_requirements_query(template.title)
    print(f"Query: {query}")

    try:
        result = app.invoke({
            "question": query,
            "template_fields": template_fields
        })
        generation = result.get("generation", "")
        action = result.get("crag_action", "unknown")
        print(f"CRAG Action: {action.upper()}")

        # ESTRAZIONE FONTI: Recupera i doc validati dal grafo
        k_in = result.get("k_in", [])
        k_ex = result.get("k_ex", [])
        all_docs = k_in + k_ex

        sources = list({
            Path(d.metadata.get("source", "unknown")).name for d in all_docs if d.metadata.get("source")})

    except Exception as e:
        print(f"Errore CRAG: {e}")
        generation = ""
        sources = []

    is_abstention = not generation or ABSTENTION_MSG in generation

    section = TemplateSection(
        title = template.title,
        level = 2,
        content = "",
        section_type = "ready_markdown"
    )

    return CompiledDocument(
        template = template,
        sections = [CompiledSection(
            section = section,
            generated_content = generation if not is_abstention else "",
            is_abstention = is_abstention,
            sources = sources
        )]
    )