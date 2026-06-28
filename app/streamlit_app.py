# ============================================================
# app/streamlit_app.py
# Phase 3: Chat UI
# ============================================================
# WHY STREAMLIT:
# Fastest way to turn a Python script into a demoable web app —
# zero frontend code needed, perfect for a portfolio project demo
# you can link directly in your resume/README.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.rag_chain import ask

st.set_page_config(page_title="Research Paper RAG Chatbot", page_icon="📄")

st.title("📄 Research Paper Q&A Chatbot")
st.caption("Ask questions about the PDFs you've ingested — powered by Mistral (Groq) + Chroma + LangChain")

# WHY SESSION STATE:
# Streamlit reruns the whole script on every interaction. Without
# session_state, chat history would reset on every new message.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.caption(s["snippet"])

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = ask(question)
            except Exception as e:
                answer, sources = f"⚠️ Error: {e}", []

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.caption(s["snippet"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)**:
        1. Your question is embedded into a vector
        2. The most relevant chunks from your PDFs are retrieved
        3. Those chunks + your question are sent to **Mistral** (via Groq)
        4. The model answers grounded in your documents only

        **Stack:** LangChain · Chroma · HuggingFace Embeddings · Groq API
        """
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
