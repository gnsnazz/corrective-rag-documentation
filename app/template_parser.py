import re
from pathlib import Path
from dataclasses import dataclass, field

# Soglia minima di campi per considerare una tabella rilevante
MIN_FIELDS = 2

# Keyword istruttive per PASS 3 di _extract_instructions
INSTRUCTION_KEYWORDS = [
    "describe", "list", "enter", "specify", "fill",
    "add", "update", "document", "state", "note",
]

# Colonne che identificano una tabella di mapping standard (IEC/ISO → sezione)
# Queste tabelle non sono mai la tabella target da compilare
MAPPING_TABLE_SIGNALS = ["document section", "document sections"]

@dataclass
class TemplateSection:
    """Singola sezione di un template."""
    title: str
    level: int                                                 # 1 = #, 2 = ##, 3 = ###
    content: str                                               # testo grezzo della sezione
    section_type: str                                          # "table", "prose", "mixed"
    table_headers: list[str] = field(default_factory = list)   # colonne se tabella
    instructions: str = ""                                     # istruzioni per il generatore


@dataclass
class ParsedTemplate:
    """Template completo parsato."""
    file_name: str
    file_path: str
    title: str                                                  # titolo principale (primo heading)
    sections: list[TemplateSection] = field(default_factory = list)
    target_table_section: TemplateSection | None = None         # sezione con la tabella da compilare


def _detect_section_type(content: str) -> tuple[str, list[str]]:
    """
    Rileva il tipo di sezione ed eventualmente le colonne della tabella.
    """
    table_lines = [l for l in content.split("\n") if "|" in l and l.strip().startswith("|")]
    has_table = len(table_lines) >= 2

    prose_lines = [
        l for l in content.split("\n")
        if l.strip()
        and not l.strip().startswith("|")
        and not l.strip().startswith("#")
        and not l.strip().startswith("---")
        and not l.strip().startswith("```")
    ]
    has_prose = len(prose_lines) >= 2 and len(prose_lines) > len(table_lines)

    table_headers = []
    if has_table:
        header_line = table_lines[0]
        table_headers = [col.strip() for col in header_line.split("|") if col.strip()]

    if has_table and has_prose:
        return "mixed", table_headers
    elif has_table:
        return "table", table_headers
    else:
        return "prose", []


def _extract_instructions(content: str) -> str:
    """
    Estrae le istruzioni dal contenuto della sezione.

    Priorità:
    1. Commenti HTML (<!-- ... -->) — istruzioni ufficiali
    2. Blockquote (> ...) — note e direttive
    3. Righe con keyword istruttive (INSTRUCTION_KEYWORDS)
    4. Placeholder tra parentesi quadre [...]
    """
    instructions = []
    seen = set()

    def _add(text: str):
        text = text.strip()
        if text and text not in seen:
            instructions.append(text)
            seen.add(text)

    # PASS 1: Commenti HTML (possono essere multi-riga)
    html_comments = re.findall(r"<!--(.*?)-->", content, re.DOTALL)
    for comment in html_comments:
        _add(comment.strip())

    # PASS 2: Blockquote
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

    # PASS 3: Righe con keyword istruttive
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith("<!--"):
            continue
        if any(marker in stripped.lower() for marker in INSTRUCTION_KEYWORDS):
            _add(stripped)

    # PASS 4: Placeholder tra parentesi quadre
    for line in content.split("\n"):
        stripped = line.strip()
        if re.search(r"\[.*?]", stripped) and not stripped.startswith("|"):
            _add(stripped)

    return "\n".join(instructions) if instructions else content.strip()


def _title_from_filename(path: Path) -> str:
    stem = path.stem
    cleaned = stem.replace("-", " ").replace("_", " ")
    return cleaned.title()


def _is_document_title(heading_text: str) -> bool:
    if re.match(r"^\d+[.)]\s+", heading_text):
        return False
    return True


def _resolve_title(headings: list[tuple], path: Path) -> str:
    if not headings:
        return _title_from_filename(path)

    first_line, first_level, first_text = headings[0]

    if first_level == 1 and _is_document_title(first_text):
        return first_text

    for _, level, text in headings[:3]:
        if level == 1 and _is_document_title(text):
            return text

    return _title_from_filename(path)


def _is_mapping_table(headers: list[str]) -> bool:
    """
    Le tabelle di mapping standard OpenRegulatory (IEC/ISO → Document Section)
    hanno sempre una colonna che contiene "document section".
    Queste non sono mai la tabella target da compilare.
    """
    normalized = [h.lower() for h in headers]
    return any(signal in h for h in normalized for signal in MAPPING_TABLE_SIGNALS)


def _count_table_fields(section: TemplateSection) -> int:
    """
    Conta i campi significativi di una sezione-tabella.
    Funziona sia per tabelle normali (colonne = campi) sia per tabelle trasposte
    (righe = campi, come nel template Bug Fixes).
    """
    lines = [l.strip() for l in section.content.split("\n") if l.strip().startswith("|")]
    if len(lines) < 2:
        return 0

    rows = [[c.strip() for c in line.split("|") if c.strip()] for line in lines]
    # Rimuove righe separatore (es. |---|---|)
    rows = [r for r in rows if not all(set(cell) <= set("-: ") for cell in r)]
    if not rows:
        return 0

    num_cols = max(len(r) for r in rows)
    num_rows = len(rows)

    # Tabella normale: i campi sono le colonne (header row)
    # Tabella trasposta: i campi sono le righe (prima colonna = nome campo)
    # Prendiamo il massimo come stima del numero di campi distinti
    return max(num_cols, num_rows)


def find_target_section(template: ParsedTemplate) -> TemplateSection | None:
    """
    Identifica la sezione che contiene la tabella da compilare.

    Logica:
    1. Considera solo sezioni di tipo "table" o "mixed"
    2. Scarta le tabelle di mapping standard (IEC/ISO → Document Section)
    3. Scarta tabelle con troppo pochi campi (MIN_FIELDS)
    4. Tra i candidati, prende quella con più campi
    """
    candidates = []

    for section in template.sections:
        if section.section_type not in ("table", "mixed"):
            continue
        if not section.table_headers:
            continue
        if _is_mapping_table(section.table_headers):
            continue

        n_fields = _count_table_fields(section)
        if n_fields <= MIN_FIELDS:
            continue

        candidates.append((n_fields, section))

    if not candidates:
        return None

    # Prende il candidato con più campi
    candidates.sort(key = lambda x: x[0], reverse = True)
    return candidates[0][1]


def parse_template(template_path: str) -> ParsedTemplate:
    """
    Legge un template .md e lo splitta in sezioni strutturate.
    Imposta automaticamente target_table_section sulla sezione da compilare.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template non trovato: {template_path}")

    text = path.read_text(encoding = "utf-8")
    lines = text.split("\n")

    headings = []
    for i, line in enumerate(lines):
        match = re.match(r"^(#{1,4})\s+(.+)", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((i, level, title))

    main_title = _resolve_title(headings, path)

    sections = []
    for idx, (line_num, level, title) in enumerate(headings):
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

    parsed = ParsedTemplate(
        file_name = path.name,
        file_path = str(path),
        title = main_title,
        sections = sections
    )

    # Identifica la sezione target e la salva nel template
    parsed.target_table_section = find_target_section(parsed)

    return parsed


def extract_template_fields(template: ParsedTemplate) -> list[str]:
    """
    Estrae i nomi dei campi dalla tabella target del template.

    Gestisce due layout:
    - Tabella normale: colonne = campi, usa la prima riga
    - Tabella trasposta: righe = campi, usa la prima colonna

    Restituisce lista vuota se non è stata trovata una tabella target valida.
    """
    section = template.target_table_section
    if section is None:
        return []

    lines = [l.strip() for l in section.content.split("\n") if l.strip().startswith("|")]
    if len(lines) < 2:
        return []

    rows = [[c.strip() for c in line.split("|") if c.strip()] for line in lines]
    rows = [r for r in rows if not all(set(cell) <= set("-: ") for cell in r)]
    if not rows:
        return []

    num_rows = len(rows)
    num_cols = max(len(r) for r in rows)

    # Tabella normale: più colonne che righe → i campi sono gli header (prima riga)
    # Tabella trasposta: più righe che colonne → i campi sono la prima colonna
    if num_cols >= num_rows:
        fields = rows[0]
    else:
        fields = [r[0] for r in rows if len(r) >= 2]

    return fields if len(fields) > MIN_FIELDS else []


def parse_all_templates(template_paths: list[str]) -> list[ParsedTemplate]:
    """Parsa tutti i template nella lista e stampa un report di debug."""
    parsed = []
    for path in template_paths:
        try:
            template = parse_template(path)
            target = template.target_table_section
            print(f"  Parsato: {template.file_name} → {len(template.sections)} sezioni")
            for s in template.sections:
                marker = " <- TARGET" if s is target else ""
                print(f"    [{s.section_type:5s}] {s.title}{marker}")
                if s.table_headers:
                    print(f"           colonne: {s.table_headers}")
            if target:
                fields = extract_template_fields(template)
                print(f"  Campi estratti: {fields}")
            else:
                print(f"  ATTENZIONE: nessuna tabella target trovata in {template.file_name}")
            parsed.append(template)
        except Exception as e:
            print(f"  ERRORE parsando {path}: {e}")
    return parsed