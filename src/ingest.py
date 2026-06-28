# ============================================================
# src/ingest.py
# Phase 1: Document Ingestion Pipeline
# ============================================================
# WHY THIS FILE EXISTS:
# RAG can't retrieve from nothing. This script runs ONCE (or
# whenever you add new PDFs) to convert raw documents into a
# searchable vector store.
#
# Pipeline: PDF -> raw text -> chunks -> embeddings -> Chroma DB
# ============================================================

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── CONFIG ───────────────────────────────────────────────────
DATA_DIR = "data/papers"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents(data_dir: str):
    """
    STEP 1 — LOAD
    WHY: PDFs aren't plain text. PyPDFLoader extracts text page by
    page and keeps metadata (source filename, page number) attached
    to each chunk — this is what lets us cite sources later.
    """
    print(f"[1/4] Loading PDFs from '{data_dir}'...")
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"      Loaded {len(documents)} pages from PDFs.")
    return documents


def split_documents(documents):
    """
    STEP 2 — CHUNK
    WHY CHUNKING MATTERS:
    - LLMs have context limits; you can't stuff a whole paper in.
    - Embedding a full paper as ONE vector loses precision — a
      32-page paper embedded as one vector is "about everything
      and nothing." Smaller chunks = sharper retrieval.
    - OVERLAP (100 chars) prevents losing context at chunk
      boundaries — e.g., a sentence split across two chunks still
      has some shared context in both.

    WHY RecursiveCharacterTextSplitter specifically:
    It tries to split on paragraph breaks first, then sentences,
    then words — preserving semantic structure instead of
    chopping mid-sentence like a naive fixed-length splitter.
    """
    print(f"[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"      Created {len(chunks)} chunks.")
    return chunks


def embed_and_store(chunks, persist_dir: str, embedding_model: str):
    """
    STEP 3 — EMBED
    WHY bge-small-en-v1.5:
    - Runs locally (no API cost, no rate limits during ingestion).
    - 384-dim embeddings — small but strong relative to size,
      a known efficient choice for retrieval tasks.
    - Keeps your project free to run end-to-end except the LLM call.

    STEP 4 — STORE IN CHROMA
    WHY CHROMA:
    - Persists to disk automatically (persist_directory) — no
      need to re-embed every time you restart the app.
    - Stores embeddings + original text + metadata together, so
      retrieval returns ready-to-use context AND source info.
    """
    print(f"[3/4] Loading embedding model '{embedding_model}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # needed for cosine similarity
    )

    print(f"[4/4] Embedding {len(chunks)} chunks and storing in Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"      Done. Vector store saved to '{persist_dir}/'.")
    return vectorstore


def main():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(
            f"No PDFs found in '{DATA_DIR}'. Add at least one PDF before running ingestion."
        )

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    embed_and_store(chunks, CHROMA_DIR, EMBEDDING_MODEL)

    print("\n✅ Ingestion complete. You can now run the RAG chain or the Streamlit app.")


if __name__ == "__main__":
    main()
