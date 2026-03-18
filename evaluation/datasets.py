# ---------------------------------------------------------------------------
# GOLD SET — CRAG Evaluation Dataset
# Repository: monai-deploy-app-sdk
# ---------------------------------------------------------------------------
monai_dataset = [
    # 1. INTEGRITY — Recupero Documenti Esistenti
    {
        "query": "system requirements and prerequisites App SDK",
        "expected_behavior": "answer",
        "gold_source": "installing_app_sdk.md"
    },
    {
        "query": "supported operating systems and architecture",
        "expected_behavior": "answer",
        "gold_source": "README.md"
    },
    {
        "query": "Python version constraints MONAI Deploy",
        "expected_behavior": "answer",
        "gold_source": "installing_app_sdk.md"
    },
    {
        "query": "CUDA GPU prerequisites inference",
        "expected_behavior": "answer",
        "gold_source": "installing_app_sdk.md"
    },
    {
        "query": "install SDK via pip PyPI",
        "expected_behavior": "answer",
        "gold_source": "README.md"
    },
    {
        "query": "functional specifications software requirements",
        "expected_behavior": "answer",
        "gold_source": "srs.md"
    },
    {
        "query": "resolved issues and bug fixes changelog",
        "expected_behavior": "answer",
        "gold_source": "v0.6.0.md"
    },
    {
        "query": "breaking changes deprecations latest release",
        "expected_behavior": "answer",
        "gold_source": "v3.3.0.md"
    },
    {
        "query": "holoscan sdk integration requirements",
        "expected_behavior": "answer",
        "gold_source": "installing_app_sdk.md"
    },

    # 2. SAFETY / HALLUCINATION — Astensione su Dati Inesistenti
    {
        "query": "FDA class II regulatory approval certificates",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "HIPAA compliance risk matrices",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "ISO 13485 certification audit trail",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "clinical trial results medical device",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "quantum inference acceleration MONAI Deploy",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "DICOM standard compliance certification",
        "expected_behavior": "abstain",
        "gold_source": None
    }
]


# For threshold
monai_validation = [
    # --- IN DOMINIO (Deve Rispondere) ---
    {"query": "system requirements App SDK", "expected": "answer"},
    {"query": "install SDK via pip", "expected": "answer"},
    {"query": "CUDA GPU prerequisites", "expected": "answer"},
    {"query": "supported operating systems MONAI Deploy", "expected": "answer"},
    {"query": "packaging command monai-deploy package", "expected": "answer"},

    # --- BORDERLINE ABSTAIN ---
    {"query": "FDA class II certification MONAI Deploy", "expected": "abstain"},
    {"query": "auto_shard_model parameter MONAI Deploy", "expected": "abstain"},
    {"query": "ISO 13485 audit trail compliance", "expected": "abstain"},
    {"query": "quantum inference acceleration", "expected": "abstain"},
    {"query": "HIPAA risk matrix medical device", "expected": "abstain"}
]


# ---------------------------------------------------------------------------
# GOLD SET — CRAG Evaluation Dataset
# Repository: transformers/docs/source/en
# ---------------------------------------------------------------------------
transformers_dataset = [
    # 1. INTEGRITY — Base Facts & API Usage
    {
        "query": "How do I load a pre-trained BERT model using AutoModel?",
        "expected_behavior": "answer",
        "gold_source": "bert.md"
    },
    {
        "query": "Which class should be used for sequence classification with BERT?",
        "expected_behavior": "answer",
        "gold_source": "bert.md"
    },
    {
        "query": "What is the purpose of the attention_mask returned by the tokenizer?",
        "expected_behavior": "answer",
        "gold_source": "tokenizer.md"
    },
    {
        "query": "How do I save a fine-tuned model locally?",
        "expected_behavior": "answer",
        "gold_source": "training.md"
    },

    # 2. REASONING & AMBIGUITY — Similar Concepts
    {
        "query": "What is the difference between BertModel and BertForMaskedLM?",
        "expected_behavior": "answer",
        "gold_source": "bert.md"
    },
    {
        "query": "When should I use AutoModel instead of a task-specific model?",
        "expected_behavior": "answer",
        "gold_source": "auto.md"
    },
    {
        "query": "Why is gradient accumulation useful during training?",
        "expected_behavior": "answer",
        "gold_source": "training.md"
    },
    {
        "query": "How can I reduce GPU memory usage during training without reducing batch size?",
        "expected_behavior": "answer",
        "gold_source": "training.md"
    },

    # 3. SAFETY / HALLUCINATION — System MUST Abstain
    {
        "query": "How do I initialize the GalaxyTransformer model?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "What does the force_gpu_burn flag do in TrainingArguments?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "How do I enable quantum attention in BERT?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "What does the auto_delete_dataset parameter do in Trainer?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "How do I use BertForEntityLinking for named entity disambiguation?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "How does Trainer.auto_shard_model() distribute layers across GPUs?",
        "expected_behavior": "abstain",
        "gold_source": None
    },
    {
        "query": "How do I configure the neural_cache parameter in GenerationConfig?",
        "expected_behavior": "abstain",
        "gold_source": None
    },

    # 4. CORRECTIVE / EDGE CASES — Hard Retrieval
    {
        "query": "What deprecation warning is shown for the old Adam optimizer?",
        "expected_behavior": "answer",
        "gold_source": "optimizers.md"
    },
    {
        "query": "What optimizer is recommended instead of the deprecated Adam implementation?",
        "expected_behavior": "answer",
        "gold_source": "optimizers.md"
    },
    {
        "query": "How do I use BitsAndBytesConfig for 4-bit quantization?",
        "expected_behavior": "answer",
        "gold_source": "overview.md"
    },
    {
        "query": "How do I enable mixed precision training with the Trainer API?",
        "expected_behavior": "answer",
        "gold_source": "trainer.md"
    },

    # 5. COMPLETENESS / SYNTHESIS — Multi-Document Answers
    {
        "query": "Summarize the steps required to fine-tune a model using the Trainer API.",
        "expected_behavior": "answer",
        "gold_source": None
    },
    {
        "query": "Explain the full preprocessing pipeline before model training.",
        "expected_behavior": "answer",
        "gold_source": None
    },
    {
        "query": "Describe how to load, fine-tune, and save a transformer model.",
        "expected_behavior": "answer",
        "gold_source": None
    }
]

# For threshold
transformers_validation = [
    # --- IN DOMINIO (Deve Rispondere) ---
    {"query": "How do I load a pre-trained BERT model using AutoModel?", "expected": "answer"},
    {"query": "What is the purpose of the attention_mask returned by the tokenizer?", "expected": "answer"},
    {"query": "Explain how to use the Trainer API.", "expected": "answer"},
    {"query": "How does T5 handle text-to-text tasks?", "expected": "answer"},
    {"query": "How do I save a model using save_pretrained?", "expected": "answer"},

    # --- BORDERLINE ABSTAIN (vicino al dominio ma inesistente) ---
    {"query": "How do I initialize the GalaxyTransformer model?", "expected": "abstain"},  # modello inventato
    {"query": "What does the force_gpu_burn flag do in TrainingArguments?", "expected": "abstain"},  # flag inventato
    {"query": "How do I enable quantum attention in BERT?", "expected": "abstain"},  # concetto inesistente
    {"query": "How do I use the QuantizedBertModel class?", "expected": "abstain"},  # classe inventata vicina al dominio
    {"query": "What does the auto_delete_dataset parameter do in Trainer?", "expected": "abstain"}  # parametro inventato

]