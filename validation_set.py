validation_queries = [
    # --- DOMANDE IN DOMINIO (Deve Rispondere) ---
    {"query": "How do I load a pre-trained BERT model using AutoModel?", "expected": "answer"},
    {"query": "Which class should be used for sequence classification with BERT?", "expected": "answer"},
    {"query": "What is the purpose of the attention_mask returned by the tokenizer?", "expected": "answer"},
    {"query": "How do I save a fine-tuned model locally?", "expected": "answer"},
    {"query": "Explain how to use the Trainer API.", "expected": "answer"},

    # --- DOMANDE OUT-OF-DOMAIN O INESISTENTI (Deve Astenersi/Cercare) ---
    {"query": "How do I initialize the GalaxyTransformer model?", "expected": "abstain"},  # Modello inventato
    {"query": "What is the recipe for a good pizza?", "expected": "abstain"},  # Fuori dominio netto
    {"query": "What does the force_gpu_burn flag do in TrainingArguments?", "expected": "abstain"},  # Flag inventato
    {"query": "How do I enable quantum attention in BERT?", "expected": "abstain"},  # Concetto inesistente
    {"query": "Who won the World Cup in 2022?", "expected": "abstain"},  # Fuori dominio
]