# 📄 RAG-Powered Research Paper Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about PDF documents (research papers, reports) using **only the content of those documents** — grounded answers, no hallucination, with source citations.

> Ask a question → relevant chunks are retrieved from your PDFs → an LLM generates an answer based strictly on that context → you see the answer **and** which document/page it came from.

---

## 🧱 Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Mistral-7B (via **Groq API**) | Fast, free-tier inference on an open-weight model |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local) | Free, runs locally, no API cost for embedding |
| Vector Store | **Chroma** | Persistent, lightweight, metadata-aware retrieval |
| Orchestration | **LangChain** (LCEL) | Industry-standard RAG pipeline composition |
| UI | **Streamlit** | Fast, demoable chat interface |
| Evaluation | **RAGAS** | Objective metrics: faithfulness, relevancy, precision, recall |

---

## 🏗️ Architecture

```
PDF documents
     │
     ▼
[ingest.py]  Load → Chunk (800 chars, 100 overlap) → Embed → Store in Chroma
     │
     ▼
Chroma Vector DB (persisted locally)
     │
     ▼
[rag_chain.py]  User question → embed → retrieve top-k chunks
     │                                         │
     ▼                                         ▼
Prompt (context + question) ──────────▶ Mistral-7B (Groq)
     │
     ▼
Answer + cited sources
     │
     ▼
[streamlit_app.py]  Chat UI
```

---

## 📂 Project Structure

```
rag-chatbot/
├── data/papers/          # Place your PDFs here
├── src/
│   ├── ingest.py         # Document ingestion pipeline
│   ├── rag_chain.py      # Retrieval + generation logic
│   └── evaluate.py       # RAGAS evaluation
├── app/
│   └── streamlit_app.py  # Chat UI
├── chroma_db/             # Auto-created vector store (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Usage

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key
```bash
cp .env.example .env
# Edit .env and add your free Groq API key from https://console.groq.com
```

### 3. Add documents
Place PDF files in `data/papers/`.

### 4. Run ingestion (one-time, or whenever you add new PDFs)
```bash
python src/ingest.py
```

### 5. Launch the chatbot
```bash
streamlit run app/streamlit_app.py
```

### 6. (Optional) Run evaluation
Edit the `test_questions` list in `src/evaluate.py` to match your documents, then:
```bash
python src/evaluate.py
```

---

## 📊 Evaluation Results

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | _run evaluate.py_ | Are answers grounded in retrieved context (no hallucination)? |
| Answer Relevancy | _run evaluate.py_ | Does the answer address the actual question? |
| Context Precision | _run evaluate.py_ | Are retrieved chunks actually relevant? |
| Context Recall | _run evaluate.py_ | Did retrieval find everything needed? |

*(Fill this table in with your actual RAGAS scores after running evaluation on your documents.)*

---

## 🔑 Key Design Decisions

- **Chunk size 800 / overlap 100** — balances context completeness vs. retrieval precision.
- **Local embeddings (bge-small)** — keeps the pipeline free and fast; only the final generation step calls an external API.
- **Temperature = 0** for generation — factual Q&A should be deterministic and grounded, not creative.
- **Source citation in the UI** — every answer shows which document/page it came from, for trust and verifiability.

---

## 🛣️ Possible Extensions

- Swap Chroma for Pinecone/Weaviate for cloud-scale deployment
- Add conversational memory (multi-turn follow-up questions)
- Add re-ranking (e.g. cross-encoder) after initial retrieval
- Support more file types (docx, web pages, markdown)

---

## 📜 License
MIT
