"""Retrieval and answer generation, shared by the Streamlit app and the command line.

Command-line chat:  python rag.py
"""
import os
import warnings

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore", message="Direct use of automatic function calling.*")

EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "enterprise_documents"

TOP_K = 3
RELEVANCE_THRESHOLD = 0.5
HISTORY_TURNS = 3  # question/answer pairs passed to the model
FALLBACK_ANSWER = "I could not find this information in the enterprise document."


def load_components():
    """Create the vector store and LLM clients. Call once and reuse."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    return vectorstore, llm


def retrieve_documents(vectorstore, question, k=TOP_K, threshold=RELEVANCE_THRESHOLD):
    """Return the top-k chunks whose relevance score clears the threshold."""
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    return [document for document, score in results if score >= threshold]


def format_history(history, turns=HISTORY_TURNS):
    """history is a list of (question, answer) pairs; keep only the most recent."""
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-turns:])


def build_prompt(question, documents, history_text):
    context = "\n\n".join(document.page_content for document in documents)
    return f"""
You are an enterprise knowledge assistant.

Answer the question using ONLY the document context below.
Use the previous conversation only to understand follow-up
questions and references such as "it" or "that".

If the answer is not present in the document context, say exactly:
"{FALLBACK_ANSWER}"

Do not invent information. Give a clear and concise answer.

Previous conversation:
{history_text}

Document context:
{context}

Current question:
{question}

Answer:
"""


def extract_text(response):
    """Gemini can return either a string or a list of content parts."""
    if isinstance(response.content, list):
        return "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return response.content


def source_pages(documents):
    """Unique 1-based page numbers, in retrieval order."""
    pages = []
    for document in documents:
        page = document.metadata.get("page")
        if isinstance(page, int):
            page += 1
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def answer_question(vectorstore, llm, question, history):
    """Return (answer, source_pages) for one question."""
    history_text = format_history(history)

    # Include the previous question so follow-ups like "give an example" retrieve well.
    retrieval_query = f"{history[-1][0]}\n{question}" if history else question
    documents = retrieve_documents(vectorstore, retrieval_query)

    # Nothing relevant enough: skip the LLM call and return the fixed fallback.
    if not documents:
        return FALLBACK_ANSWER, []

    answer = extract_text(llm.invoke(build_prompt(question, documents, history_text)))

    if FALLBACK_ANSWER in answer:
        return FALLBACK_ANSWER, []
    return answer, source_pages(documents)


def main():
    vectorstore, llm = load_components()
    history = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "bye"}:
            print("\nGoodbye!")
            break
        if not question:
            continue
        answer, pages = answer_question(vectorstore, llm, question, history)
        print(f"\nAssistant: {answer}")
        if pages:
            print("Sources: " + ", ".join(f"Page {p}" for p in pages))
        history.append((question, answer))


if __name__ == "__main__":
    main()
