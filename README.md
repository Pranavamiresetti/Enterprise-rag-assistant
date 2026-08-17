# 📚 Enterprise Knowledge Assistant

### Retrieval-Augmented Generation for Enterprise Document Question Answering

An AI-powered Enterprise Knowledge Assistant that enables users to ask natural-language questions about enterprise documents and receive grounded answers using **Retrieval-Augmented Generation (RAG)**.

The system combines semantic retrieval with **Google Gemini** to retrieve relevant document context from **ChromaDB** and generate concise, context-aware responses with source-page references.

---

## 🚀 Overview

Enterprise documents often contain large amounts of domain-specific information that can be difficult to search manually.

This project implements a complete **Retrieval-Augmented Generation pipeline** that allows users to interact with enterprise knowledge through a conversational interface.

Instead of directly asking the language model to generate an answer, the system:

1. Processes the enterprise document.
2. Splits the document into retrievable chunks.
3. Generates vector embeddings.
4. Stores the embeddings in ChromaDB.
5. Retrieves the most relevant chunks for a user query.
6. Provides the retrieved context to Gemini.
7. Generates a grounded response.
8. Displays the source pages used for the response.

---

## 🏗️ System Architecture

```text
                    ┌────────────────────────┐
                    │    Enterprise PDF      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Document Processing &  │
                    │      Chunking           │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Gemini Embeddings    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │       ChromaDB          │
                    │    Vector Database      │
                    └────────────┬───────────┘
                                 │
                                 │
          ┌──────────────────────▼─────────────────────┐
          │                                             │
          │               User Question                 │
          │                                             │
          └──────────────────────┬─────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Semantic Retrieval     │
                    │       Top-K = 3        │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Relevant Document       │
                    │        Context          │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │      Gemini LLM         │
                    │  Context-Grounded QA    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Answer + Source Pages  │
                    └────────────────────────┘

---

## ✨ Key Features

- 📄 **Enterprise Document Processing**
  - Processes enterprise PDF documents into searchable knowledge chunks.

- 🧩 **Semantic Chunk Retrieval**
  - Retrieves the most relevant document sections for each user query.

- 🧠 **Google Gemini Integration**
  - Uses Gemini for context-grounded natural-language question answering.

- 🔎 **Vector Similarity Search**
  - Stores and retrieves document embeddings using ChromaDB.

- 💬 **Conversational Question Answering**
  - Maintains recent conversation context to support follow-up questions.

- 📚 **Source Attribution**
  - Displays the document pages used to generate each response.

- 🐳 **Dockerized Deployment**
  - The application can be built and executed using Docker and Docker Compose.

- 🔐 **Environment-Based API Configuration**
  - API credentials are loaded through environment variables rather than hard-coded in the application.

---
```
---
## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM | Google Gemini |
| Embeddings | Google Gemini Embeddings |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| Document Processing | PDF / LangChain |
| Containerization | Docker |
| Container Orchestration | Docker Compose |
| Environment Management | python-dotenv |
| Version Control | Git & GitHub |
```
---

## 📁 Project Structure

```text
Enterprise-rag-assistant/
│
├── data/
│   └── company_policy.pdf
│
├── app.py
├── ingest.py
├── rag.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── .dockerignore
├── .gitignore
└── README.md
```

### 📌 File Responsibilities

| File / Directory | Purpose |
|---|---|
| `app.py` | Streamlit user interface and conversational interaction |
| `ingest.py` | Processes the PDF and creates document embeddings |
| `rag.py` | Handles document retrieval and RAG operations |
| `data/` | Contains the enterprise knowledge document |
| `Dockerfile` | Defines the application container image |
| `docker-compose.yml` | Simplifies containerized deployment |
| `requirements.txt` | Lists Python dependencies |
| `.env` | Stores API credentials locally |
| `.gitignore` | Prevents sensitive and unnecessary files from being committed |

> **Note:** `.env` is intentionally excluded from version control and should never be committed to GitHub.

---

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pranavamiresetti/Enterprise-rag-assistant.git
cd Enterprise-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv enterprise-rag
```

Activate the environment:

**macOS / Linux**
```bash
source enterprise-rag/bin/activate
```

**Windows**
```bash
enterprise-rag\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

> **Security:** Never commit your `.env` file or expose your API key publicly.

### 5. Build the Vector Database

Run the ingestion pipeline:

```bash
python ingest.py
```

This processes the enterprise PDF, generates embeddings using Google Gemini Embeddings, and stores them in ChromaDB.

### 6. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🐳 Docker Deployment

The application also includes Docker configuration for containerized deployment.

### Build and Run with Docker Compose

```bash
docker compose up --build
```

After the containers start, open the Streamlit application in your browser.

To stop the application:

```bash
docker compose down
```

---
---

## 💬 Usage

Once the application is running, open the Streamlit interface in your browser.

Enter a natural-language question related to the enterprise document.

### Example Questions

```text
What is the company's leave policy?

What are the eligibility requirements mentioned in the document?

What benefits are provided to employees?

What is the policy regarding working hours?

Summarize the key points of the company policy.
```

The assistant retrieves the most relevant document chunks from ChromaDB and provides a context-grounded response using Google Gemini.

Each response also displays the **source document pages** used during retrieval, improving transparency and traceability.

---

## 🔐 Security

- API credentials are stored using environment variables.
- `.env` is excluded from Git version control.
- API keys should never be hard-coded in application source code.
- Sensitive configuration files should not be committed to the repository.

---

## ⚠️ Limitations

- Responses are grounded only in the indexed enterprise document.
- The quality of answers depends on the quality and coverage of the source document.
- The current implementation uses a single enterprise knowledge document.
- Retrieval performance depends on the selected chunking and similarity-search configuration.

---

## 🔮 Future Improvements

- Support multiple enterprise documents and document collections.
- Implement authentication and role-based access control.
- Add conversation persistence across sessions.
- Introduce retrieval evaluation metrics.
- Add monitoring and logging for production deployments.
- Deploy the application to a cloud platform.

---

## 👨‍💻 Author

**Pranav Amiresetti**

M.Tech – Data Analytics  
National Institute of Technology, Jalandhar

[GitHub](https://github.com/Pranavamiresetti)

---

## 📄 License

This project is intended for educational and portfolio purposes.
