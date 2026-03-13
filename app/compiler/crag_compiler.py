from app.crag.graph import build_crag_graph
from app.template_parser import ParsedTemplate, TemplateSection
from app.recomposer import CompiledDocument, CompiledSection
from app.config import format_source, ABSTENTION_MSG

# Sezione fittizia: non parsata dal template, è la tabella completa generata dall'LLM.
CRAG_SECTION_TYPE = "ready_markdown"

def build_requirements_query(repo_name: str, template_title: str) -> str:
    """
    Costruisce la macro-query per recuperare tutti i chunk rilevanti dal vector store per un dato template.
    """
    return (f"{repo_name} {template_title} "
            f"software requirements features capabilities dependencies configuration specifications")


def process_crag_compliance(template: ParsedTemplate, template_fields: list[str], repo_name: str) -> CompiledDocument:
    """
    Compila il template con approccio single-pass:
    1. Una sola macro-query al CRAG per recuperare tutti i chunk rilevanti
    2. Il grader filtra i documenti rilevanti
    3. Il nodo generate produce l'intera tabella dei requisiti
    """
    print(f"Avvio Compilazione CRAG Single-Pass - {template.title}")

    app = build_crag_graph()
    query = build_requirements_query(repo_name ,template.title)
    print(f"Query: {query}")

    generation = ""
    context = ""
    sources = []

    try:
        result = app.invoke({
            "question": query,
            "template_fields": template_fields
        })

        generation = result.get("generation", "")
        context = result.get("context", "")
        action = result.get("crag_action", "unknown")
        print(f"CRAG Action: {action.upper()}")

        # ESTRAZIONE FONTI: Recupera i doc validati dal grafo
        k_in = result.get("k_in", [])
        k_ex = result.get("k_ex", [])

        seen = set()
        for d in k_in + k_ex:
            raw_source = d.metadata.get("source", "")
            if raw_source:
                normalized = format_source(raw_source)
                if normalized not in seen:
                    seen.add(normalized)
                    sources.append(normalized)

    except Exception as e:
        print(f"Errore CRAG: {e}")

    is_abstention = not generation or ABSTENTION_MSG in generation

    # Usa il titolo della sezione target, non il titolo del documento
    target = template.target_table_section
    section_title = target.title if target else template.title

    section = TemplateSection(
        title = section_title,
        level = 2,
        content = "",
        section_type = CRAG_SECTION_TYPE
    )

    return CompiledDocument(
        template = template,
        sections = [CompiledSection(
            section = section,
            generated_content = generation if not is_abstention else "",
            is_abstention = is_abstention,
            sources = sources,
            context = context
        )]
    )