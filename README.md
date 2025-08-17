
# Legal Q\&A Assistant (RAG + OpenAI)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **OpenAI GPT models** to answer legal questions based on a set of uploaded documents.
The system retrieves the most relevant text chunks using **embeddings** and then generates precise, context-aware answers.

---

## Features

* 📄 **Document Ingestion** – Upload and process legal documents (PDF, DOCX, or text).
* 🧩 **Chunking & Embeddings** – Breaks documents into chunks and generates vector embeddings using `text-embedding-3-large`.
* 🔍 **Vector Search** – Finds the most relevant passages for a user’s query.
* 🤖 **Answer Generation** – Uses `gpt-4o-mini` to generate clear, legally-aware responses.
* 💾 **Persistent Storage** – Vector database is stored in `vectorstore.pkl` for quick retrieval.

---

## Tech Stack

* **Language**: Python
* **Libraries**: OpenAI, FAISS / pickle, python-dotenv
* **Models**:

  * Chat Model → `gpt-4o-mini`
  * Embeddings Model → `text-embedding-3-large`

---

## How It Works

1. **Load Environment Variables** from `.env` (API keys + model names).
2. **Ingest Documents** → split into chunks with overlap for better context.
3. **Embed & Store** → embeddings saved locally (`vectorstore.pkl`).
4. **Ask a Question** → retrieve top chunks matching the query.
5. **Generate Answer** → OpenAI GPT refines the answer with supporting text.

---

## Outcomes

* ✅ Built a functional **legal assistant** capable of answering legal queries with reference to the provided documents.
* ✅ Successfully tested with multiple legal clauses (e.g., **confidentiality agreements, NDAs, contracts**) and observed accurate contextual responses.
* ✅ Designed with **scalability in mind** (can expand to more documents, more models, or even a web interface).

---

##  Challenges Faced

* 🔑 **Environment Setup** → Managing `.env` and API keys properly in VS Code.
* 📦 **Dependencies** → Missing modules (`dotenv`) and ensuring smooth installation across different machines.
* 💾 **File Handling** → Handling `vectorstore.pkl` errors when the file did not exist initially.
* 🔄 **GitHub Sync** → Issues with remote commits (other contributors’ commits appearing), resolved by force-pushing a clean commit history.
* 🧠 **RAG Accuracy** → Needed careful tuning of chunk size and overlap to maintain context without overwhelming the model.

---

## Next Steps

* [ ] Add a **web UI** (Streamlit or Flask) for easier use.
* [ ] Support **more file formats** (Excel, scanned PDFs with OCR).
* [ ] Enhance **evaluation metrics** to measure accuracy of responses.
* [ ] Deploy as a **web service** for broader accessibility.

---

## 👤 Author

**Urbanus Peter Kathitu ([@KathituCodes](https://github.com/KathituCodes))**


