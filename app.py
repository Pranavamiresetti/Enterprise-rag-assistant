import streamlit as st

from rag import answer_question, load_components


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
# 3. LOAD RAG COMPONENTS (cached, so created once per server)
# =========================================================

@st.cache_resource
def get_components():
    return load_components()


try:
    vectorstore, llm = get_components()
except ValueError as error:
    st.error(str(error))
    st.stop()


# =========================================================
# 4. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 5. SIDEBAR
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
# 6. HEADER
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
# 7. WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Ask me anything about the enterprise knowledge document."
    )


# =========================================================
# 8. DISPLAY CHAT HISTORY
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


# =========================================================
# 9. USER INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about the document..."
)


# =========================================================
# 10. HANDLE USER QUESTION
# =========================================================

if question:

    # Turn stored messages into (question, answer) pairs
    messages = st.session_state.messages
    history = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
    ]

    with st.spinner("Searching the document..."):
        answer, sources = answer_question(vectorstore, llm, question, history)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    st.rerun()
