# Corrective-RAG (CRAG)

Implementation of **Corrective Retrieval Augmented Generation** built with **LangGraph**.

## Overview

Standard RAG systems retrieve documents and feed them directly to a language model, trusting that the retrieved context is relevant. In practice, retrievers often return noisy or irrelevant documents, leading to hallucinated or low-quality answers.

CRAG addresses this by introducing a **self-correcting loop** between retrieval and generation. Each retrieved document is evaluated by an LLM grader that classifies it as *correct*, *ambiguous*, or *incorrect*. Based on the aggregate confidence across all documents, the system takes one of three actions:

- **CORRECT** — the retrieved documents contain a clear answer: generate directly
- **AMBIGUOUS** — partial evidence found: rewrite the query, run a corrective retrieval round, and re-evaluate
- **INCORRECT** — no useful evidence: discard everything, rewrite the query, and search again from scratch

If after all retrieval rounds the system still has no valid evidence, it **abstains** rather than generating a potentially hallucinated answer. This precision-first approach ensures that every generated response is grounded in verified context.

The system also applies **knowledge refinement** on ambiguous documents: each document is split into fine-grained strips, filtered by semantic similarity to the query, and recomposed — keeping only the relevant portions and discarding noise.

---

## Project Structure

```
CRAG/
├── app/
│   ├── __init__.py
│   ├── config.py                # Global config, thresholds, constants
│   ├── embeddings.py            # Embedding model
│   ├── ingest_documents.py      # Document ingestion into Chroma
│   ├── utils.py                 # Utilities (save output, logging)
│   └── crag/
│       ├── __init__.py
│       ├── graph.py             # LangGraph workflow + timed_node wrapper
│       ├── models.py            # LLM, grader, vectorstore, retriever, strip splitter
│       ├── nodes.py             # Node functions (retrieve, grade, generate, ...)
│       ├── prompts.py           # All prompt templates
│       └── state.py             # GraphState + CragDocument definition
│
├── evaluation/
│   ├── __init__.py
│   ├── judge.py                 # LLM judge for automated scoring
│   ├── run_evaluation.py        # Full evaluation pipeline + metrics report
│   ├── optimize_threshold.py    # Confidence threshold grid search
│   ├── validation_set.py        # Quick validation queries
│   └── crag_metrics.csv         # Latest evaluation results
│
├── data/
│   ├── transformers/            # HuggingFace Transformers docs (source corpus)
│   └── vectorstore/             # Chroma DB persistence
│
├── output_docs/                 # Generated documentation output
├── main.py                      # Single-query entry point
├── requirements.txt
├── .env                         # API keys
└── .gitignore
```

---

## Setup

```bash
pip install -r requirements.txt
```

`.env`:
```
API_KEY=your-key-here
```

**Ingest corpus:**
```bash
python -m app.ingest_documents
```

**Run single query:**
```bash
python main.py
```

**Run evaluation:**
```bash
python -m evaluation.run_evaluation
```

---

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Routing | Balanced Accuracy | Avg of answer accuracy and abstain accuracy |
| Routing | Precision / Recall / F1 | Binary classification (answer vs abstain) |
| Generation | Faithfulness (1-5) | Is the answer grounded in retrieved context? |
| Generation | Relevance (1-5) | Does the answer address the query? |
| Retrieval | Recall@k Pre/Post | Was the gold document retrieved / retained after grading? |
| Performance | Per-node latency | Timing breakdown by pipeline stage |