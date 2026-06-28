# ============================================================
# src/rag_chain.py
# Phase 2: Retrieval + Generation Chain
# ============================================================
# WHY THIS FILE EXISTS:
# This is the "online" part of RAG — runs every time a user asks
# a question. Takes a query -> retrieves relevant chunks ->
# builds a prompt -> calls the LLM -> returns answer + sources.
# ============================================================

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 4

# NOTE: Groq deprecates/updates hosted model names periodically.
# Check https://console.groq.com/docs/models for the current
# Mistral/Mixtral model slug before running this.


PROMPT_TEMPLATE = """You are a helpful research assistant. Answer the question
using ONLY the context below. If the answer isn't in the context, say
"I don't have enough information in the provided documents to answer that."
Do not make up information.

Important: Paraphrase the context in your own words. Do not copy exact
sentences or phrases from the context, and do not use quotation marks or
bracketed citations within your answer text.

Context:
{context}

Question: {question}

Answer (be concise and grounded strictly in the context above):"""


def format_docs(docs):
    """
    WHY: The retriever returns LangChain Document objects with
    metadata. We need to flatten them into plain text for the
    prompt, while keeping track of sources separately for citation.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_retriever(k: int = TOP_K):
    """
    WHY A SEPARATE RETRIEVER FUNCTION:
    Keeps vector-store loading logic isolated so both the chain
    and the evaluation script (Phase 4) can reuse it identically.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    # similarity search returns the k most relevant chunks by
    # cosine distance between query embedding and stored embeddings
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_rag_chain():
    """
    WHY LCEL (LangChain Expression Language) PIPE SYNTAX:
    This declarative style (using `|`) makes the data flow
    explicit and is the current LangChain standard — interviewers
    familiar with LangChain will recognize this pattern immediately.

    Flow:
      question
        -> retriever fetches docs (context)
        -> prompt template fills in {context} and {question}
        -> LLM generates the answer
        -> output parser extracts plain string
    """
    retriever = get_retriever()

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,  # WHY 0: factual QA needs deterministic,
                         # grounded answers — not creative variation
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def ask(question: str):
    """
    Convenience wrapper used by the Streamlit app: returns both
    the generated answer AND the source chunks (for citation UI).
    """
    rag_chain, retriever = build_rag_chain()

    # Run retrieval separately too, so we can show sources in the UI
    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200] + "...",
        }
        for doc in source_docs
    ]
    return answer, sources


if __name__ == "__main__":
    # quick manual test from the command line
    q = input("Ask a question about your documents: ")
    answer, sources = ask(q)
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for s in sources:
        print(f"- {s['source']} (page {s['page']})")
