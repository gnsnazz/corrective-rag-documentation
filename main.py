from langchain_ollama import ChatOllama
from app.config import OLLAMA_MODEL

def main():
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0
    )

    print(llm.invoke("Test modello"))

if __name__ == "__main__":
    main()
