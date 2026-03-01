from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- JUDGE LLM ---
llm_judge = ChatAnthropic(
    model_name  = "claude-haiku-4-5-20251001",
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)

# --- SCORE SCHEMA ---
class LLMScore(BaseModel):
    faithfulness: int = Field(description = "Score 1-5: Is the answer grounded in the context? (1=Hallucination, 5=Fully Supported)")
    answer_relevance: int = Field(description = "Score 1-5: Does the answer address the user query? (1=Irrelevant, 5=Perfect)")
    reasoning: str = Field(description = "Short reasoning for the scores")

# --- JUDGE CHAIN ---
parser = JsonOutputParser(pydantic_object = LLMScore)

judge_chain = PromptTemplate(
template = """You are an evaluator. Rate the RAG system output.

QUERY: {query}
RETRIEVED CONTEXT: {context}
SYSTEM ANSWER: {answer}

Provide scores (1-5) for:
1. Faithfulness: Is the answer derived ONLY from the context? (1=Hallucination, 5=Fully Supported)
2. Answer Relevance: Is the answer useful for the query? (1=Irrelevant, 5=Perfect)

{format_instructions}""",
    input_variables = ["query", "context", "answer"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
) | llm_judge | parser


def evaluate_with_llm(query: str, context: str, answer: str) -> LLMScore:
    """Valuta una risposta usando il judge LLM."""
    try:
        res = judge_chain.invoke({"query": query, "context": context, "answer": answer})
        return LLMScore(**res)
    except Exception as e:
        print(f"  [WARN] LLM Judge error: {e}")
        return LLMScore(faithfulness = 1, answer_relevance = 1, reasoning = "Error")