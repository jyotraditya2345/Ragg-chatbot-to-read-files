# Ragg-chatbot-to-read-files
# 🧠 PDF Knowledge Base Chatbot (Gemini + ChromaDB)

An AI-powered chatbot that reads your PDFs, stores them in ChromaDB, and answers questions using Google Gemini with context-aware responses. Built with Gradio for easy interaction.

---

## 🚀 Features

- 📄 Upload PDFs and extract content automatically
- 🔍 Split text into token-based chunks with overlap
- 🧠 Store & retrieve document chunks using ChromaDB
- 🤖 Use Gemini API to answer questions contextually
- 💬 Gradio UI for chatting with your documents
- 🔐 Basic authentication for secure access

---

## 📁 Project Structure

📁 data/ → Place your PDF files here
📄 app.py → Main application with Gradio UI
📄 chromadb_utils.py → Chunking + ChromaDB integration
📄 utils.py → PDF reading and file gathering
📄 requirements.txt → Project dependencies

---

## 💻 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>

2. Install dependencies
pip install -r requirements.txt

3. Add your PDFs
Put any .pdf files inside the data/ directory.

4. Run the app
python app.py

By default, it runs on:
http://0.0.0.0:4065

Credentials:

Username: Quantum123

Password: pass123

🧠 How It Works
PDFs are read using LangChain’s PyPDFLoader.

Text is tokenized into overlapping chunks.

Chunks are embedded and stored in ChromaDB.

On question, most relevant chunk is retrieved.

Gemini API uses the chunk as context to generate a smart response.

📌 To-Do
 Add file upload via Gradio

 Optimize repeated embeddings

 Support more file formats (.txt, .docx)

 Add local LLM support (for offline mode)

