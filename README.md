# Integrating Corrective-RAG for Automated Compliance Documentation

Automatic generation of regulatory compliance documentation from a GitHub repository, using **Corrective RAG (CRAG)** built with **LangGraph**.

The system processes a target repository and fills [OpenRegulatory](https://openregulatory.com) templates for IEC 62304 / ISO 13485 compliance. The case study corpus is **MONAI Deploy App SDK**.

---

## Overview

Two distinct pipelines handle different input types:

**Case 1 — Bug Fixes Documentation List (structured data)**
GitHub Issues are fetched via the GitHub API and compiled directly into the template. Each issue becomes one row in the output table. No semantic retrieval — the data is already structured.

**Case 2 — Software Requirements List (unstructured data)**
A full CRAG pipeline retrieves relevant chunks from the embedded repository documentation, grades and refines them, and generates the requirements table in a single pass.

---

## Pipeline

### Shared steps

**Corpus ingestion** (`ingest_documents.py`) — run once before the first execution. Loads all `.md` and `.rst` files from the target repository, splits them into chunks, embeds them with a local sentence-transformers model, and persists the vector store to Chroma.

**Template parsing** (`template_parser.py`) — reads the OpenRegulatory `.md` template, splits it into sections, detects the target table (filtering out standard mapping tables), and extracts the field names to be filled.

### Case 1 — Bug Fixes (structured data)

1. `github_fetcher.py` calls the GitHub Issues API and downloads all closed issues labeled `bug`, including comments
2. `direct_compiler.py` iterates over each issue, formats it as structured text, and calls the LLM to produce a `| Field | Value |` table for that issue
3. `recomposer.py` assembles all per-issue tables into a single transposed Markdown table (fields as columns, bugs as rows) and saves the final report

### Case 2 — Software Requirements (unstructured data)

1. `crag_compiler.py` builds a macro-query from the template title and sends it to the CRAG graph
2. The CRAG graph executes:
   - **Retrieve** — fetch the top-k most similar chunks from Chroma
   - **Grade** — an LLM evaluator classifies each chunk as `correct`, `ambiguous`, or `incorrect`
   - **Route** based on cumulative confidence score:
     - `CORRECT` (conf ≥ upper threshold) → generate directly
     - `AMBIGUOUS` (lower ≤ conf < upper) → rewrite query, run corrective retrieval with MMR, re-grade
     - `INCORRECT` (conf < lower) → discard internal knowledge, rewrite, search from scratch
   - **Knowledge Refinement** — ambiguous documents are split into fine-grained strips, filtered by cosine similarity, and recomposed to keep only relevant portions
   - **Generate** — produce the full requirements table from verified context only
   - **Abstain** — if no valid evidence is found after all retries, return an abstention message instead of hallucinating
3. `recomposer.py` wraps the generated table with document metadata and saves the final report

---

## Project Structure

```
CRAG/
├── app/
│   ├── compiler/
│   │   ├── direct_compiler.py   # Case 1: structured data pipeline
│   │   └── crag_compiler.py     # Case 2: CRAG single-pass pipeline
│   ├── crag/
│   │   ├── graph.py             # LangGraph workflow
│   │   ├── models.py            # LLM, grader, vectorstore, retriever
│   │   ├── nodes.py             # Node functions (retrieve, grade, generate, ...)
│   │   ├── prompts.py           # Prompt templates
│   │   └── state.py             # GraphState + CragDocument
│   ├── retriever/
│   │   ├── github_fetcher.py    # GitHub Issues API client
│   │   └── ingest_documents.py  # Document ingestion into Chroma
│   ├── config.py                # Thresholds, paths, constants
│   ├── embeddings.py            # Embedding model
│   ├── recomposer.py            # Assembles the final Markdown report
│   └── template_parser.py       # Parses OpenRegulatory .md templates
│
├── evaluation/
│   ├── datasets.py              # Gold sets for MONAI and Transformers corpora
│   ├── judge.py                 # LLM judge (faithfulness, relevance, table scoring)
│   ├── run_evaluation.py        # CRAG routing + retrieval + generation evaluation
│   ├── evaluate_table.py        # Table completeness / correctness / hallucination
│   └── optimize_threshold.py    # Confidence threshold grid search
│
├── data/
│   ├── monai/                   # MONAI Deploy App SDK source corpus
│   ├── templates/               # OpenRegulatory .md templates
│   └── vectorstore/             # Chroma DB persistence
│
├── reports/                     # Generated compliance reports (output)
├── main.py                      # Entry point
├── requirements.txt
├── docker-compose.yml
└── .env
```

---

## Setup

### With Docker

**1. Configure environment**
```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and GITHUB_TOKEN to .env
```

**2. Build the image**
```bash
docker compose build
```

**3. Ingest the corpus** (only needed once, or after corpus changes)
```bash
docker compose run --rm crag python -m app.retriever.ingest_documents
```

**4. Fetch GitHub Issues** (Case 1 only)
```bash
docker compose run --rm crag python app/retriever/github_fetcher.py
```

### Without Docker

**1. Configure environment**
```bash
cp .env.example .env
# Add ANTHROPIC_API_KEY and GITHUB_TOKEN to .env
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Ingest the corpus**
```bash
python -m app.retriever.ingest_documents
```

**4. Fetch GitHub Issues** (Case 1 only)
```bash
python app/retriever/github_fetcher.py
```

---

## Usage

### With Docker
```bash
docker compose run --rm crag python main.py
```

### Without Docker
```bash
python main.py
```

Interactive menu:
```
TEMPLATE COMPILER
Seleziona il caso:
  1 - Bug Fixes
  2 - Software Requirements
Scelta: _
```

Or pass the case directly:
```bash
# With Docker
docker compose run --rm crag python main.py bugs
docker compose run --rm crag python main.py requirements

# Without Docker
python main.py bugs
python main.py requirements
```

Reports are saved in `reports/`.

---

## Evaluation

### With Docker
```bash
docker compose run --rm crag python -m evaluation.run_evaluation
docker compose run --rm crag python -m evaluation.evaluate_table
docker compose run --rm crag python -m evaluation.optimize_threshold
```

### Without Docker
```bash
python -m evaluation.run_evaluation
python -m evaluation.evaluate_table
python -m evaluation.optimize_threshold
```

| Category | Metric | Description |
|----------|--------|-------------|
| Routing | Balanced Accuracy | Avg of answer accuracy and abstain accuracy |
| Routing | Precision / Recall / F1 | Binary classification (answer vs abstain) |
| Generation | Faithfulness (1–5) | Answer grounded in retrieved context |
| Generation | Relevance (1–5) | Answer addresses the query |
| Retrieval | Recall@k Pre/Post | Gold document retrieved / retained after grading |
| Performance | Per-node latency | Timing breakdown by pipeline stage |