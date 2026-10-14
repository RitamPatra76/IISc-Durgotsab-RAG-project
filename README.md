# IISc-Durgotsab-RAG-project
# How to Run

Follow these quick steps to run the **IISc Sharodiya Durgotsab — Graph RAG** project.

---

## 1. Prerequisites

- **Python 3.10+**
- An **OpenAI API Key** or **OpenRouter API Key**

---

## 2. Setup

### 1. Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd iisc-durgotsab-graph-rag
```

### 2. Create and activate a virtual environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Configure API Key

Create a `.env` file in the root folder (or copy from `.env.example`):
```bash
cp .env.example .env
```

Open `.env` and add your API key:
```env
OPENAI_API_KEY="your_openai_api_key_here"

# OR (for free/open-weight models via OpenRouter):
# OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

---

## 4. Run the Q&A Assistant

Run the interactive Question Answering assistant:
```bash
python graph_rag.py
```

To jump directly into interactive mode:
```bash
python graph_rag.py --interactive
```

Type your questions in English or Bengali (বাংলা), and type `quit` to exit.
