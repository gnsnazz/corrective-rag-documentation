# Corrective-RAG (CRAG)

Questa repository contiene un’implementazione di **Corrective Retrieval Augmented Generation (CRAG)** basata su **LangGraph**.

L’obiettivo del progetto è dimostrare come costruire un sistema RAG **robusto agli errori di retrieval**, capace di:

* valutare criticamente le evidenze recuperate
* correggere dinamicamente il retrieval quando necessario
* evitare hallucinations quando la conoscenza non è disponibile

---

## Cos’è CRAG (in breve)

CRAG estende il paradigma RAG introducendo una fase di **valutazione e correzione della conoscenza** prima della generazione.

Pipeline concettuale del paper:

1. **Retrieve** – recupero documenti
2. **Evaluate** – classificazione dei documenti (correct / ambiguous / incorrect)
3. **Knowledge Correction**

   * **Knowledge Refinement** (stripping)
   * **Knowledge Searching** (query rewrite + nuova ricerca)
4. **Generate** – generazione condizionata dal tipo di evidenza

La generazione finale dipende da:

* **k_in** → conoscenza valida iniziale
* **k_ex** → conoscenza ottenuta tramite ricerca correttiva

---

## Organizzazione Progetto

```
CRAG/
├── app/
│ ├── crag/
│ │ ├── __init__.py
│ │ ├── graph.py          # Definizione del grafo LangGraph (flusso CRAG)
│ │ ├── nodes.py          # Implementazione dei nodi CRAG
│ │ ├── prompts.py        # Prompt LLM
│ │ └── state.py          # Definizione dello stato globale (GraphState, CragDocument)
│ │
│ ├── __init__.py
│ ├── config.py           # Configurazioni globali
│ ├── embeddings.py       # Modello di embedding
│ ├── ingest_documents.py # Script di ingestione documenti nel vectorstore
│ └── utils.py            # Utility (salvataggio output, logging)
│
├── data/
│ ├── transformers/       # Repository Hugging Face Transformers
│ └── vectorstore/        # Persistenza Chroma DB
│
├── output_docs/          # Output generato (documentazione CRAG)
├── .env                  # Variabili ambiente (API keys)
├── .gitignore
├── main.py               # Entry point
├── rag_test.py           # Script di test RAG base
└── README.md             # Readme progetto
```


