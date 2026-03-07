import re
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class TemplateSection:
    """Singola sezione di un template."""
    title: str
    level: int                                                # 1 = #, 2 = ##, 3 = ###
    content: str                                              # testo grezzo della sezione
    section_type: str                                         # "table", "prose", "mixed"
    table_headers: list[str] = field(default_factory = list)  # colonne se tabella
    instructions: str = ""                                    # istruzioni per il generatore


@dataclass
class ParsedTemplate:
    """Template completo parsato."""
    file_name: str
    file_path: str
    title: str                                                 # titolo principale (primo heading)
    sections: list[TemplateSection] = field(default_factory = list)
    target_table_section: TemplateSection | None = None


def _detect_section_type(content: str) -> tuple[str, list[str]]:
    """
    Rileva il tipo di sezione ed eventualmente le colonne della tabella.
    """
    # Cerca tabelle markdown (righe con |)
    table_lines = [l for l in content.split("\n") if "|" in l and l.strip().startswith("|")]
    has_table = len(table_lines) >= 2  # header + separator minimo

    # Cerca prosa (righe di testo non vuote, non tabella, non heading)
    prose_lines = [
        l for l in content.split("\n")
        if l.strip()
        and not l.strip().startswith("|")
        and not l.strip().startswith("#")
        and not l.strip().startswith("---")
        and not l.strip().startswith("```")
    ]
    has_prose = len(prose_lines) >= 2

    # Estrae headers tabella
    table_headers = []
    if has_table:
        header_line = table_lines[0]
        table_headers = [
            col.strip()
            for col in header_line.split("|")
            if col.strip()
        ]

    # Classifica
    if has_table and has_prose:
        return "mixed", table_headers
    elif has_table:
        return "table", table_headers
    else:
        return "prose", []


def _extract_instructions(content: str) -> str:
    """
    Estrae le istruzioni dal contenuto della sezione.

    Priorità di estrazione:
    1. Commenti HTML (<!-- ... -->) — istruzioni ufficiali
    2. Blockquote (> ...) — usati per note e direttive
    3. Righe con keyword istruttive (describe, list, enter, ...)
    4. Placeholder tra parentesi quadre [...]
    """
    instructions = []
    seen = set()  # evita duplicati

    def _add(text: str):
        text = text.strip()
        if text and text not in seen:
            instructions.append(text)
            seen.add(text)

    # --- PASS 1: Commenti HTML (possono essere multi-riga) ---
    html_comments = re.findall(r"<!--(.*?)-->", content, re.DOTALL)
    for comment in html_comments:
        _add(comment.strip())

    # --- PASS 2: Blockquote (intere righe che iniziano con >) ---
    blockquote_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">"):
            blockquote_lines.append(stripped.lstrip("> ").strip())
        else:
            if blockquote_lines:
                _add(" ".join(blockquote_lines))
                blockquote_lines = []
    if blockquote_lines:
        _add(" ".join(blockquote_lines))

    # --- PASS 3: Righe con keyword istruttive ---
    for line in content.split("\n"):
        stripped = line.strip()

        # Salta righe già catturate come commenti/blockquote
        if stripped.startswith(">") or stripped.startswith("<!--"):
            continue

        if any(marker in stripped.lower() for marker in [
            "describe", "list", "enter", "specify", "fill",
            "add", "update", "document", "state", "note",
        ]):
            _add(stripped)

    # --- PASS 4: Placeholder tra parentesi quadre ---
    for line in content.split("\n"):
        stripped = line.strip()
        if re.search(r"\[.*?]", stripped) and not stripped.startswith("|"):
            _add(stripped)

    return "\n".join(instructions) if instructions else content.strip()


def _title_from_filename(path: Path) -> str:
    """
    Genera un titolo pulito dal nome del file.
    """
    stem = path.stem
    # Sostituisci - e _ con spazi, capitalizza
    cleaned = stem.replace("-", " ").replace("_", " ")
    return cleaned.title()


def _is_document_title(heading_text: str) -> bool:
    """
    Controlla se un heading sembra un titolo di documento (non una sezione numerata).
    """
    # Se inizia con numero + punto, è una sezione numerata
    if re.match(r"^\d+[.)]\s+", heading_text):
        return False
    return True


def _resolve_title(headings: list[tuple], path: Path) -> str:
    """
    Determina il titolo del documento con logica robusta.

    Strategia:
    1. Se c'è un H1 che sembra un titolo → usa quello
    2. Se il primo heading è numerato o non c'è H1 → usa il nome file
    """
    if not headings:
        return _title_from_filename(path)

    first_line, first_level, first_text = headings[0]

    # Caso ideale: H1 con titolo vero
    if first_level == 1 and _is_document_title(first_text):
        return first_text

    # Cerca un H1 tra i primi heading (a volte c'è metadata prima)
    for _, level, text in headings[:3]:
        if level == 1 and _is_document_title(text):
            return text

    # Nessun H1 valido trovato → fallback al nome file
    return _title_from_filename(path)


def parse_template(template_path: str) -> ParsedTemplate:
    """
    Legge un template .md e lo splitta in sezioni strutturate.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template non trovato: {template_path}")

    text = path.read_text(encoding = "utf-8")
    lines = text.split("\n")

    # Trova tutti gli heading e le loro posizioni
    headings = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,4})\s+(.+)", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((i, level, title))

    # Titolo principale: cerca H1, altrimenti fallback al nome file pulito
    main_title = _resolve_title(headings, path)

    # Splitta il contenuto in sezioni (tra un heading e il successivo)
    sections = []
    for idx, (line_num, level, title) in enumerate(headings):
        # Contenuto = da questo heading al prossimo (o fine file)
        start = line_num + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()

        if not content:
            continue

        section_type, table_headers = _detect_section_type(content)
        instructions = _extract_instructions(content)

        sections.append(TemplateSection(
            title = title,
            level = level,
            content = content,
            section_type = section_type,
            table_headers = table_headers,
            instructions = instructions
        ))

    return ParsedTemplate(
        file_name = path.name,
        file_path = str(path),
        title = main_title,
        sections = sections
    )


def extract_template_fields(template: ParsedTemplate) -> list[str]:
    """
    Estrae i campi dal template rilevando automaticamente
    il tipo di tabella (row-fields o column-fields).
    """
    best_fields = []

    for section in template.sections:
        if section.section_type not in ["table", "mixed"]:
            continue

        lines = [l.strip() for l in section.content.split("\n") if l.strip().startswith("|")]

        if len(lines) < 2:
            continue

        rows = [ [c.strip() for c in line.split("|") if c.strip()] for line in lines ]
        num_rows = len(rows)
        num_cols = max(len(r) for r in rows)

        # CASE 1: Row-fields table
        if num_rows > num_cols:
            fields = []
            for r in rows:
                if len(r) < 2:
                    continue

                field = r[0]

                if set(field) == {"-", ":"}:
                    continue

                fields.append(field)

            #if len(fields) > 1:
                #fields = fields[1:]

        # CASE 2: Column-fields table
        else:
            fields = rows[0]

        if len(fields) > len(best_fields):
            best_fields = fields

    return best_fields


def parse_all_templates(template_paths: list[str]) -> list[ParsedTemplate]:
    """Parsa tutti i template nella lista."""
    parsed = []
    for path in template_paths:
        try:
            template = parse_template(path)
            print(f"  Parsato: {template.file_name} -> {len(template.sections)} sezioni")
            for s in template.sections:
                print(f"    [{s.section_type:5s}] {s.title}")
                if s.table_headers:
                    print(f"           colonne: {s.table_headers}")
            parsed.append(template)
        except Exception as e:
            print(f"  ERRORE parsando {path}: {e}")
    return parsed
