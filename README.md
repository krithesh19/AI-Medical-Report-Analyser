# 🧠 AI Medical Report Analyser – RAG-powered Clinical Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-orange)](https://www.pinecone.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini--1.5--Flash-red)](https://ai.google.dev/gemini-api)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-pink)](https://streamlit.io/)
[![CI](https://github.com/krithesh19/AI-Medical-Report-Analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/krithesh19/AI-Medical-Report-Analyser/actions)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Live App:** https://ai-medical-report-analyser-krit19.streamlit.app

> ⚠️ *Disclaimer: This project is for educational use only. It is not a medical diagnostic tool.*

---

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Objectives](#-objectives)
- [Methodology](#-methodology)
- [Demo](#demo)
- [Evaluation](#-evaluation)
- [Workflow](#-workflow)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [My Contribution](#-my-contribution)
- [Installation](#-installation)
- [Docker](#-docker)
- [Future Scope](#-future-scope)
- [Authors](#-authors)
- [References](#-references)

---

## 📖 Introduction

Healthcare generates vast amounts of unstructured clinical data. Interpreting **patient medical reports** can be time-consuming and confusing for non-specialists. Large Language Models (LLMs) are powerful at text synthesis but prone to **hallucination** when used without grounding.

This project builds a **Retrieval-Augmented Generation (RAG)** assistant that ensures **trustworthy, cited answers** to patient queries. By embedding both **reference helper documents** and **patient PDFs**, the system retrieves relevant evidence before passing it to an LLM (Gemini 1.5 Flash). The result is a **transparent and educational chatbot** that can summarise, interpret, and explain medical reports.

---

## 🎯 Objectives

- Provide **faithful, source-grounded answers** based on patient data
- Ensure **traceability** by citing whether information came from `[patient]`, `[helpbook]`, or `[web]`
- Minimise hallucinations by enforcing retrieval-first prompting
- Build a **user-friendly Streamlit app** for uploads, chat, and metrics
- Support **evaluation metrics** (faithfulness, helpfulness, latency, hallucination rate)

---

## 🔎 Methodology

### 1. Document Ingestion
- **Helper Docs**: Uploaded once (from `/helperDocs`) and stored persistently
- **Patient Reports**: Uploaded per session (PDF/TXT), split into 1000-char chunks with 150 overlap, tagged with `session_id`

### 2. Embedding & Storage
- Embeddings: `MiniLM-L6-v2 (384-dim)` from HuggingFace
- Storage: **Pinecone** vector database with two indexes:
  - `GENERAL_INDEX` → helper/reference medical docs
  - `PATIENT_INDEX` → per-session patient documents

### 3. Retrieval
- Dual retriever using **Maximal Marginal Relevance (MMR)**:
  - Helpbook: k=6, λ=0.2
  - Patient: k=10, λ=0.35 (filtered by `session_id`)
- If retrieval fails → automatic fallback to **DuckDuckGo** web search

### 4. RAG Generation
- **Prompt template** enforces patient-first grounding and inline citations
- LLM: **Gemini 1.5 Flash** with `temperature=0.2`
- Conversation memory (`RunnableWithMessageHistory`) maintains multi-turn context

### 5. Output
- Streamlit chat displays concise 2–3 paragraph answers
- Each response includes key values, reference ranges, and inline citations

---

## Demo

![Full page](readmePics/full_page.png)
![Chat section](readmePics/full_chat.png)

---

## 📊 Evaluation

The system tracks metrics per session:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | % of answers supported by retrieved context |
| **Hallucination rate** | % of answers unsupported by context |
| **Grounding rate** | % of answers citing `[patient]` or `[helpbook]` |
| **Retrieval latency** | Time to fetch from Pinecone (ms) |
| **LLM latency** | Time for Gemini to generate (ms) |

📈 Example chart:
![Evaluation Metrics](readmePics/evaluation_metrics.png)

---

## 🖥️ Workflow

![Workflow Diagram](readmePics/workflow_horizontal1.png)

Patient PDF / TXT
│
▼
Document Ingestion
(1000-char chunks, 150 overlap, session_id tag)
│
▼
HuggingFace Embeddings (MiniLM-L6-v2, 384-dim)
│
├──────────────────────────────────┐
▼                                  ▼
PATIENT_INDEX (Pinecone)        GENERAL_INDEX (Pinecone)
session_id filter               medical reference docs
MMR k=10, λ=0.35               MMR k=6, λ=0.2
│                                  │
└──────────┬───────────────────────┘
▼
Merged Context
[patient] + [helpbook]
│
▼ (fallback if empty)
DuckDuckGo Search [web]
│
▼
Gemini 1.5 Flash (temp=0.2)
+ RunnableWithMessageHistory
│
▼
Cited Answer + Evaluation Metrics

---

## ✨ Features

- Upload & process **helper documents** once — persisted across sessions
- Upload **patient reports** per session (PDF or TXT)
- Three LangChain tools:
  - `RAG_QA` — answer questions with inline citations
  - `Summarise Patient Report` — concise report overview
  - `Interpret Lab Test` — test-level explanations with reference ranges
- **DuckDuckGo fallback** with explicit `[web]` labels
- **Session isolation** — answers never mix across different patients
- Automatic metrics logging to `session_metrics.csv`

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Flash |
| RAG Framework | LangChain 0.3 |
| Vector Database | Pinecone (dual-index) |
| Embeddings | HuggingFace MiniLM-L6-v2 (384-dim) |
| Retrieval Strategy | Maximal Marginal Relevance (MMR) |
| Conversation Memory | RunnableWithMessageHistory |
| Fallback Search | DuckDuckGo API |
| UI | Streamlit |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## 👨‍💻 My Contribution (Kritheshvar Vinothkumar)

This is a group project. My core responsibilities were the **RAG pipeline, LLM integration, and evaluation framework**:

| File | What I built |
|------|-------------|
| `rag.py` | Dual-retriever RAG pipeline — MMR config, context merging, session fallback logic, conversation memory |
| `llm.py` | Gemini 1.5 Flash setup — temperature tuning, grounding prompt engineering to minimise hallucination |
| `vectorstore.py` | Pinecone dual-index architecture — separate indexes for patient vs reference data |
| `agent.py` | LangChain agent with three tools: RAG QA, report summariser, lab test interpreter |
| `metrics.py` | Evaluation framework — faithfulness score, hallucination rate, latency tracking per session |

My teammate Sushmitha handled document ingestion, embeddings, and the Streamlit UI.

---

## ⚙️ Installation

```bash
git clone https://github.com/krithesh19/AI-Medical-Report-Analyser.git
cd AI-Medical-Report-Analyser

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate.ps1      # Windows

pip install -r requirements.txt
```

Set up `.env`:
```ini
PINECONE_API_KEY=your_key_here
PINECONE_REGION=us-east-1
GENERAL_INDEX_NAME=medical-helpbook
PATIENT_INDEX_NAME=patient-reports
GOOGLE_API_KEY=your_key_here
```

Run locally:
```bash
streamlit run app.py
```

---

## 🐳 Docker

```bash
docker build -t medical-analyser .
docker run -p 8501:8501 --env-file .env medical-analyser
```

---

## 📌 Future Scope

- 🌍 Multilingual and speech-enabled interface
- 🧑‍⚕️ Doctor co-pilot integrations
- 📈 Visual dashboards for retrieval quality
- 🔄 Reinforcement learning via user feedback

---

## 🧑‍💻 Authors

**Kritheshvar Vinothkumar** (24233914) — MSc Data & Computational Science, UCD
RAG pipeline · LLM integration · Evaluation framework

**Sushmitha B** (24209228)
Document ingestion · Embeddings · Streamlit UI

---

## 📚 References

1. Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS. [Link](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
2. Carbonell, J., & Goldstein, J. (1998). *The use of MMR, diversity-based reranking for reordering documents and producing summaries.* SIGIR. [Link](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)
3. Chen, Y., et al. (2025). *MRD-RAG: Enhancing medical diagnosis with multi-round RAG.* arXiv. [Link](https://arxiv.org/html/2504.07724v1)
4. Yang, R., et al. (2025). *RAG for generative AI in health care.* npj Health Systems. [Link](https://www.nature.com/articles/s44401-024-00004-1)
