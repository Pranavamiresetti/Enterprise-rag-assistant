import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0f1117;
    }

    /* Main content width */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .main-header h1 {
        font-size: 2.4rem;
        margin-bottom: 5px;
    }

    .main-header p {
        color: #9ca3af;
        font-size: 1rem;
    }

    /* Chat bubbles */
    .user-message {
        background-color: #1f2937;
        padding: 14px 18px;
        border-radius: 14px;
        margin: 10px 0;
        border: 1px solid #374151;
    }

    .assistant-message {
        background-color: #172033;
        padding: 16px 18px;
        border-radius: 14px;
        margin: 10px 0;
        border: 1px solid #26354d;
    }

    /* Source box */
    .source-box {
        background-color: #111827;
        border-left: 3px solid #6b7280;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 6px;
        color: #d1d5db;
        font-size: 0.9rem;
    }

    /* Small labels */
    .message-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #9ca3af;
        margin-bottom: 5px;
    }

    /* Divider */
    .divider {
        height: 1px;
        background-color: #252a34;
        margin: 22px 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD ENVIRONMENT
# =========================================================

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()


# =========================================================
# 4. LOAD EMBEDDING MODEL
# =========================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# =========================================================
# 5. CONNECT TO CHROMADB
# =========================================================

vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="enterprise_documents",
    embedding_function=embeddings
)


# =========================================================
# 6. CREATE RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# =========================================================
# 7. CREATE GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# =========================================================
# 8. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📚 About")

    st.write(
        "Enterprise Knowledge Assistant"
    )

    st.markdown("---")

    st.write("**Architecture**")

    st.write(
        "PDF → Chunking → Embeddings → "
        "ChromaDB → Retrieval → Gemini"
    )

    st.markdown("---")

    st.write("**Knowledge Base**")

    st.write(
        "Enterprise Expert Systems document"
    )

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 10. HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<h1>📚 Enterprise Knowledge Assistant</h1>

<p>
AI-powered question answering using Retrieval-Augmented Generation
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 11. WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Ask me anything about the enterprise knowledge document."
    )


# =========================================================
# 12. DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(message["content"])

            if message.get("sources"):

                st.caption(
                    "📄 Sources: "
                    + ", ".join(
                        f"Page {page}"
                        for page in message["sources"]
                    )
                )

        # Display sources
        if message.get("sources"):

            sources = ", ".join(
                f"Page {page}"
                for page in message["sources"]
            )

            st.markdown(
                f"""
                <div class="source-box">
                📄 <b>Sources:</b> {sources}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 13. USER INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the document..."
)


# =========================================================
# 14. PROCESS QUESTION
# =========================================================

# =========================================================
# HANDLE USER QUESTION
# =========================================================

if question:

    # Save user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # -----------------------------------------------------
    # Build conversation context
    # -----------------------------------------------------

    previous_messages = st.session_state.messages[:-1]

    conversation_history = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in previous_messages[-6:]
    )

    # -----------------------------------------------------
    # Create retrieval query
    # -----------------------------------------------------

    retrieval_query = f"""
Previous conversation:
{conversation_history}

Current question:
{question}
"""

    # -----------------------------------------------------
    # Retrieve relevant document chunks
    # -----------------------------------------------------

    documents = retriever.invoke(retrieval_query)

    # -----------------------------------------------------
    # Prepare document context
    # -----------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # -----------------------------------------------------
    # Create prompt
    # -----------------------------------------------------

    prompt = f"""
You are an enterprise knowledge assistant.

Your job is to answer questions using ONLY the
information contained in the provided enterprise document.

You should also use the previous conversation to
understand follow-up questions.

Previous conversation:
{conversation_history}

Relevant document context:
{context}

Current question:
{question}

Instructions:

1. Answer using only the provided document context.
2. Use previous conversation only to understand what
   the user is referring to.
3. If the required information is not present in the
   document context, say:
   "I could not find this information in the enterprise document."
4. Do not invent information.
5. Give a clear and concise answer.

Answer:
"""

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        answer = ""

        for item in response.content:

            if isinstance(item, dict) and item.get("type") == "text":
                answer += item.get("text", "")

    else:
        answer = response.content

    # -----------------------------------------------------
    # Get source pages
    # -----------------------------------------------------

    sources = []

    for document in documents:

        page = document.metadata.get("page")

        if isinstance(page, int):
            page = page + 1

        if page not in sources:
            sources.append(page)

    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # -----------------------------------------------------
    # Rerun to display updated conversation
    # -----------------------------------------------------

    st.rerun()