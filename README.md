# ForensicMind — Multimodal Crime Intelligence Assistant

> An AI-powered investigation dashboard that ingests FIR documents and witness statements, classifies crimes, maps applicable IPC sections, and generates structured investigation reports — all from a sleek dark tactical UI.

---

## 🔍 What It Does

- **Evidence Ingestion** — Upload FIR PDFs, witness statements (TXT), and crime scene images
- **Semantic RAG Retrieval** — ChromaDB vector store with Ollama embeddings
- **Basic RAG** — Standard semantic search → LLM-grounded answers
- **Advanced RAG** — HyDE + Multi-query expansion + Re-ranking for higher accuracy
- **Automated Crime Classification** — Keyword + evidence-based crime type detection with confidence score
- **IPC Section Mapping** — Automatically maps evidence to relevant Indian Penal Code sections (only when legally queried)
- **Structured Investigation Reports** — PDF export of complete case analysis
- **RAG Evaluation** — Precision, Recall, and MRR metrics via `/evaluate` API and UI panel

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11+ |
| **Vector Store** | ChromaDB (persistent, local) |
| **Embeddings** | Ollama (`nomic-embed-text`) |
| **LLM** | Ollama (`tinyllama` / `llama3`) |
| **Frontend** | HTML, CSS, Vanilla JS |
| **Typography** | Syne (headings), DM Sans (body) |
| **PDF Export** | jsPDF (browser-side) |
| **Icons** | Lucide Icons |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Git

### 1. Clone the repo
```bash
git clone https://github.com/navyajain30/ForensicMind-ChatBot.git
cd ForensicMind-ChatBot/crime-intelligence-assistant
```

### 2. Set up the Python environment
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Pull Ollama models
```bash
ollama pull nomic-embed-text
ollama pull tinyllama
```

### 5. Start the backend
```bash
python main.py
```

### 6. Open the app
Visit **http://localhost:8000/app** in your browser.

---

## 🔑 Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in your values.

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for fast LLM inference ([get one free](https://console.groq.com)) |
| `OLLAMA_BASE_URL` | Local Ollama server URL (default: `http://localhost:11434`) |
| `EMBEDDING_MODEL` | Ollama model for embeddings (default: `nomic-embed-text`) |
| `LLM_MODEL` | LLM model name (default: `tinyllama`) |
| `VISION_MODEL` | Vision model for image processing (default: `llava`) |

> ⚠️ **Never commit your `.env` file.** It is excluded via `.gitignore`.

---

## 📁 Project Structure

```
crime-intelligence-assistant/
├── backend/
│   ├── main.py                  # FastAPI app, API endpoints
│   ├── .env.example             # Environment variable template
│   ├── data/
│   │   └── ipc_sections.json    # Internal IPC knowledge base (35+ sections)
│   ├── rag/
│   │   ├── basic.py             # Basic RAG pipeline
│   │   ├── advanced.py          # Advanced RAG (HyDE + Multi-query + Re-ranking)
│   │   └── vector_store.py      # ChromaDB wrapper
│   ├── processors/
│   │   └── multimodal.py        # PDF, TXT, image processors
│   ├── utils/
│   │   └── llm.py               # LLM service wrapper
│   └── evaluation/
│       └── evaluate_rag.py      # Precision / Recall / MRR evaluation
└── frontend/
    ├── index.html               # App shell
    ├── css/styles.css           # Dark tactical design system
    └── js/app.js                # Frontend logic
```

---

## 📸 Screenshots

> _Upload screenshots of the dashboard, chat responses, and evidence processing here._
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/507368c2-2ce0-4d9d-9b18-6dd78c99ac80" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/72c4cf03-20b5-4780-a304-b57804ea1fd1" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/b2d84439-e3a7-4f98-954a-564b4b39d59c" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/6c5a52d5-cb22-49ee-bc03-50f992ab12b2" />

---
