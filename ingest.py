"""Ingestion pipeline: PDFs -> chunks -> Gemini embeddings -> ChromaDB.

Run:  python ingest.py     (re-running rebuilds the collection from scratch)
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIRECTORY

DATA_DIRECTORY = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment")

    # 1. Load every PDF in the data folder
    pdf_paths = sorted(Path(DATA_DIRECTORY).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIRECTORY}/")

    documents = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDF(s)")

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed and store. Drop the old collection first so re-runs do not duplicate chunks.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    ).delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    print("Documents stored in ChromaDB")


if __name__ == "__main__":
    main()
