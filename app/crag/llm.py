from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import Literal

# --- LLM ---
llm = ChatAnthropic(
    model_name = "claude-haiku-4-5-20251001", #claude-sonnet-4-5-20250929
    temperature = 0,
    timeout = None,
    stop = None,
    max_retries = 2
)

# --- GRADER ---
class Grade(BaseModel):
    """Score for relevance check."""
    score: Literal["correct", "ambiguous", "incorrect"] = Field(
        description = """Relevance classification: 'correct' (explicit answer), 'ambiguous' (needs refinement),
         or 'incorrect' (irrelevant)."""
    )

llm_grader = llm.with_structured_output(Grade)
