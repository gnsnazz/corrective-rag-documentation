from langchain_core.prompts import PromptTemplate
from app.config import ABSTENTION_MSG

# --- 1. GRADER SYSTEM MESSAGE (Strict) ---
GRADER_SYSTEM_MSG = """You are a strict technical evaluator acting as a firewall for a RAG system.
Your job is to assess the relevance of a retrieved document to a user question.

Classify the document into one of these three categories based on these STRICT abstract rules:

1. 'correct': 
   - The document contains EXPLICIT information, code, or documentation that directly answers the specific question.
   - The **Subject/Entity** requested in the question matches exactly with the one described in the document.

2. 'ambiguous': 
   - The document discusses the **Correct Subject/Entity** requested, but the specific answer is not immediately obvious, is implicit, or requires synthesizing multiple parts.
   - The document is relevant to the topic but might be noisy (e.g., logs, general discussions, or list of parameters).
   - ACTION: These documents are valuable and should be passed to the Refiner.

3. 'incorrect': 
   - **ENTITY MISMATCH**: The user asks for 'Subject A', but the document describes 'Subject B'. Even if they belong to the same domain or category, if the specific identifier/name is different, it is INCORRECT.
   - **IRRELEVANT**: The document is about a completely different topic.
   - **NOISE**: The document contains no semantic content (e.g., only imports, license headers, or empty lists).

IMPORTANT: 
- Do not assume the user made a typo. 
- If the specific Class/Function/Method name requested is not present, mark it as 'incorrect' (unless it is a generic conceptual question).
- Precision is prioritized over recall.
"""

# --- 2. REFINER PROMPT ---
# Estrae solo la polpa dai documenti ambigui
REFINE_TEMPLATE = """You are a Knowledge Refiner. 
The user asked: "{question}"

Your job is to extract RELEVANT technical content from the document snippet below.

RULES:
1. EXCLUDE NOISE: Remove conversational filler, headers, logs, or marketing fluff.
2. EXTRACT EXACT MATCHES: If the document contains the answer, keep it.
3. HANDLE CONCEPTS: If the document describes a concept SIMILAR or ALTERNATIVE to what the user asked
(e.g., user asks for 'X' but doc talks about 'Y' which serves a similar purpose), KEEP IT. This helps the system explain the difference.
4. If the document is completely unrelated (different topic), return "IRRELEVANT".

Document Snippet:
{document}"""

refine_prompt = PromptTemplate(
    template = REFINE_TEMPLATE,
    input_variables = ["question", "document"]
)

# --- 3. REWRITER PROMPT ---
# Regole severe per avere solo la stringa della query
REWRITE_TEMPLATE = """You are rewriting a failed search query.

Rules:
- Output ONLY a natural language technical query
- Do NOT use SQL, code, symbols, or explanations
- Use documentation-style keywords
- Keep it short (max 12 words)

Original query:
{question}

Previous attempts:
{history}

Rewritten query:"""

rewrite_prompt = PromptTemplate(
    template = REWRITE_TEMPLATE,
    input_variables = ["question", "history"]
)

# --- 4. GENERATOR PROMPT ---
GENERATE_PROMPT = """You are an expert technical writer compiling regulatory documentation.
Your task is to extract data from the <context> and map it EXACTLY to the requested fields.

== REQUIRED FIELDS ==
{template_fields}

== DATA (CONTEXT) ==
<context>
{context}
</context>

STRICT OUTPUT RULES:
1. FIELD REPLICATION: You MUST use EXACTLY the fields listed in "REQUIRED FIELDS" above. Do not add, remove, or rename any fields. Do not invent fields.
2. FORMAT: Output ONLY a single Markdown table with exactly TWO columns: "Field" and "Value".
3. ONE ROW PER FIELD: The generated table must have exactly one row for every field listed in the REQUIRED FIELDS.
4. MISSING DATA: If the specific data for a field is not explicitly found in the context, write exactly "N/A" in the Value column. Do not deduce or guess.
5. NO EXTRA TEXT: Do not generate introductions, summaries, or any text outside the Markdown table.

If the <context> is completely empty, return EXACTLY this string and nothing else:
"{abstention_msg}"

Table:"""

generate_prompt = PromptTemplate(
    template = GENERATE_PROMPT,
    input_variables = ["template_fields", "context"],
    partial_variables = {"abstention_msg": ABSTENTION_MSG}
)

REQUIREMENTS_GENERATE_TEMPLATE = """You are an expert technical writer compiling regulatory documentation.
Your task is to extract all software requirements from the context and structure them in a Markdown table.

== REQUIRED COLUMNS ==
{template_fields}

== CONTEXT ==
<context>
{context}
</context>

STRICT OUTPUT RULES:
1. ONE ROW PER REQUIREMENT: Each distinct software requirement becomes one row.
2. COLUMNS: Use EXACTLY the columns listed in "REQUIRED COLUMNS" as table headers.
3. MISSING DATA: MISSING DATA: If the specific value for a field is not found explicitly and verbatim in the context,
write exactly "N/A". Do not infer, deduce, or guess values under any circumstance, even for yes/no fields.
4. NO DUPLICATES: Merge duplicate requirements into one row.
5. NO EXTRA TEXT: Output ONLY the Markdown table.

If the context contains no requirements, return EXACTLY this string and nothing else:
"{abstention_msg}"
"""

requirements_generate_prompt = PromptTemplate(
    template = REQUIREMENTS_GENERATE_TEMPLATE,
    input_variables = ["context", "template_fields"],
    partial_variables = {"abstention_msg": ABSTENTION_MSG}
)