import os
import warnings

warnings.filterwarnings(

    "ignore",

    message="Direct use of automatic function calling.*"

)
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------
# 1. Load API key
# ---------------------------------

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")


# ---------------------------------
# 2. Load the same embedding model
# ---------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ---------------------------------
# 3. Connect to existing ChromaDB
# ---------------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="enterprise_documents",
    embedding_function=embeddings
)


# ---------------------------------
# 4. Similarity-based retrieval
# ---------------------------------

def retrieve_documents(question, k=3, threshold=0.5):

    results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    relevant_documents = []

    for document, score in results:

        if score >= threshold:
            relevant_documents.append(document)

    return relevant_documents

# ---------------------------------
# 5. Create Gemini LLM
# ---------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    
)

# ---------------------------------
# 6. Conversation loop
# ---------------------------------

chat_history = []

while True:

    question = input("\nYou: ")

    # Exit command
    if question.lower() in ["exit", "quit", "bye"]:
        print("\nGoodbye!")
        break

    # ---------------------------------
    # 7. Retrieve relevant documents
    # ---------------------------------

    documents = retrieve_documents(question)

    # ---------------------------------
    # 8. Prepare context
    # ---------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # ---------------------------------
    # 9. Prepare conversation history
    # ---------------------------------

    history = "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in chat_history
    )

    # ---------------------------------
    # 10. Create prompt
    # ---------------------------------

    prompt = f"""
You are an enterprise policy assistant.

Answer the user's question using ONLY the information
provided in the retrieved company policy context.

Use the conversation history to understand follow-up
questions and references such as "it", "this", "that",
"the previous concept", or "give me an example".

If the answer is not present in the company policy,
say exactly:

"I could not find this information in the company policy."

Conversation History:
{history}

Retrieved Company Policy Context:
{context}

Current User Question:
{question}

Answer:
"""

    # ---------------------------------
    # 11. Generate answer
    # ---------------------------------

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict) and item.get("type") == "text":
                answer += item.get("text", "")

    else:
        answer = response.content

    # ---------------------------------
    # 12. Display answer
    # ---------------------------------

    print("\n========================================")
    print("ANSWER")
    print("========================================")

    print(answer)

    # ---------------------------------
    # 13. Display sources
    # ---------------------------------

    print("\n========================================")
    print("SOURCES")
    print("========================================")

    if documents:

        for document in documents:

            page = document.metadata.get("page", "Unknown")

            print(
                f"- Page {page + 1 if isinstance(page, int) else page}"
            )

    else:

        print("- No relevant company policy source found.")

    # ---------------------------------
    # 14. Save conversation
    # ---------------------------------

    chat_history.append(
        (question, answer)
    )