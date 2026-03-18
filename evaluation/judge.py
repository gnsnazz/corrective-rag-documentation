from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

llm_judge = ChatAnthropic(
    model_name = "claude-haiku-4-5-20251001",
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)


class LLMScore(BaseModel):
    faithfulness: int = Field(description = "Score 1-5: Is the answer grounded in the context? (1=Hallucination, 5=Fully Supported)")
    answer_relevance: int = Field(description = "Score 1-5: Does the answer address the user query? (1=Irrelevant, 5=Perfect)")
    reasoning: str = Field(description = "Short reasoning for the scores")


_judge_structured = llm_judge.with_structured_output(LLMScore)

_JUDGE_PROMPT = """\
You are an evaluator of RAG system outputs.

QUERY:
{query}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

Rate the system answer on:
1. Faithfulness (1-5): Is the answer derived ONLY from the context? (1=Hallucination, 5=Fully Supported)
2. Answer Relevance (1-5): Is the answer useful and complete for the query? (1=Irrelevant, 5=Perfect)

Provide a short reasoning for your scores.\
"""


def evaluate_with_llm(query: str, context: str, answer: str) -> LLMScore:
    """Valuta una risposta RAG usando il judge LLM."""
    try:
        return _judge_structured.invoke([
            ("user", _JUDGE_PROMPT.format(query = query, context = context, answer = answer))
        ])
    except Exception as e:
        print(f"  [WARN] LLM Judge error: {e}")
        return LLMScore(faithfulness = 1, answer_relevance = 1, reasoning = f"Error: {e}")


class TableScore(BaseModel):
    completeness: int = Field(description = "Score 1-5: Are all template fields filled with non-N/A values where data exists? (1=Almost empty, 5=Fully filled)")
    correctness: int = Field(description = "Score 1-5: Are the extracted values grounded in the provided context? (1=Invented, 5=Fully supported)")
    hallucination:int = Field(description = "Score 1-5: Are there invented values not present in context? (1=Many hallucinations, 5=No hallucinations)")
    reasoning: str = Field(description = "Short reasoning for the scores")


_table_judge_structured = llm_judge.with_structured_output(TableScore)

_TABLE_JUDGE_PROMPT = """\
You are an expert evaluator of regulatory documentation tables.

TEMPLATE FIELDS (expected columns):
{template_fields}

RETRIEVED CONTEXT (source of truth):
{context}

GENERATED TABLE (to evaluate):
{generated_table}

Evaluate the generated table on:
1. Completeness (1-5): Are all template fields filled with meaningful values where data exists in context?
2. Correctness (1-5): Are all values directly grounded in the retrieved context?
3. Hallucination (1-5): Are there values invented by the model not present in context? (5=No hallucinations)

Provide a short reasoning for your scores.\
"""


def evaluate_table(template_fields: list[str], context: str, generated_table: str) -> TableScore:
    """Valuta la tabella generata usando il judge LLM."""
    try:
        fields_str = "\n".join(f"- {f}" for f in template_fields)
        return _table_judge_structured.invoke([
            ("user", _TABLE_JUDGE_PROMPT.format(
                template_fields = fields_str,
                context = context,
                generated_table = generated_table
            ))
        ])
    except Exception as e:
        print(f"  [WARN] Table Judge error: {e}")
        return TableScore(completeness = 1, correctness = 1, hallucination = 1, reasoning = f"Error: {e}")