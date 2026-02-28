validation_queries = [
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
    {"query": "What does the auto_delete_dataset parameter do in Trainer?", "expected": "abstain"},  # parametro inventato

]