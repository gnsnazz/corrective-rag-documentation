from langchain_ollama import OllamaLLM
from app.config import OLLAMA_MODEL

def main():
    llm = OllamaLLM(model=OLLAMA_MODEL)

    response = llm.invoke("Test modello")
    print(response)

if __name__ == "__main__":
    main()
