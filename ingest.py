import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

# Check API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "data/company_policy.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# -----------------------------
# 2. Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# 4. Store in ChromaDB
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="enterprise_documents"
)

print("Documents successfully stored in ChromaDB!")