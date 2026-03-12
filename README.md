# Agentic L1 Support PoC (CRM-less)

Python PoC for an **agentic L1 support system** (text-first, speech-ready) using **LangGraph + LangChain**.

## Quick run (summary)
- **1)** (Optional) create venv and activate:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **2)** Install deps:
  ```bash
  pip install -r requirements.txt
  ```
- **3)** (Optional, recommended) set your OpenAI key:
  ```powershell
  $env:OPENAI_API_KEY="YOUR_KEY"
  ```
- **4)** Run the CLI agent with debug:
  ```bash
  python main.py --debug
  ```

## End-to-end run (all phases)
To exercise the complete project in one flow:

1. **Environment + install**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Model / embeddings config**
   ```powershell
   $env:OPENAI_API_KEY="YOUR_KEY"
   ```
3. **Ingest KB into the vector store (Phase 3)**
   ```bash
   python scripts/ingest_kb.py
   ```
4. **Start the API server in one terminal (Phase 4)**
   ```bash
   uvicorn api:app --reload
   ```
5. **In another terminal, run the CLI agent with debug (Phase 1–2)**
   ```bash
   python main.py --debug
   ```
6. **(Optional) Enable local TTS (speech-ready)**
   ```bash
   pip install pyttsx3
   python main.py --debug --tts
   ```
7. **Run evaluations (Phase 5)**
   ```bash
   python evals/run_evals.py
   ```

## Project goals
- High-accuracy L1 troubleshooting using a Knowledge Base (RAG).
- No CRM integration (restricted requests are escalated + logged).
- Short, voice-friendly responses (speech-ready architecture).
- Session persistence during a run (LangGraph `MemorySaver`).

## Repository layout
```
.
├─ core/
│  ├─ agent.py           # LangGraph workflow
│  ├─ config.py          # env-driven settings (model, KB paths)
│  └─ state.py           # SupportState TypedDict
├─ tools/
│  ├─ knowledge_base.py  # mock KB fallback
│  ├─ rag_kb.py          # vectorstore retrieval (Chroma)
│  └─ logger.py          # restricted-action request log
├─ kb_docs/              # markdown knowledge base articles (Phase 3)
├─ scripts/
│  └─ ingest_kb.py       # builds persisted vectorstore from kb_docs/
├─ evals/
│  ├─ cases.json         # routing regression tests
│  └─ run_evals.py       # eval runner
├─ speech/
│  └─ adapters.py        # speech interface stubs + optional local TTS
├─ api.py                # FastAPI server (Phase 4)
├─ main.py               # terminal loop harness
├─ requests_log.json     # created at runtime (restricted actions)
└─ requirements.txt
```

## Architecture (all phases)
### State
`SupportState` tracks:
- `chat_history`: the conversation messages (persisted per `thread_id`)
- `current_intent`: `technical_support | restricted_action | unknown`
- `needs_human`: boolean
- `escalation`: optional structured escalation payload

### LangGraph workflow
1. **Classifier node**
   - Uses **GPT-4o** when `OPENAI_API_KEY` is set.
   - Falls back to a small heuristic router when no key is present (offline demo).
2. **Knowledge Search node**
   - Prefer **RAG** retrieval from persisted Chroma (`.vectorstore`)
   - Fallback to mock KB dictionary
   - Keeps replies concise (optimized for later speech/TTS)
3. **Escalator node**
   - For account/billing/restricted actions: explains it needs a human
   - Sets `needs_human=True`, adds `escalation` info, and logs to `requests_log.json`

### Persistence
- In-memory persistence uses LangGraph `MemorySaver`
- You keep continuity by using a consistent `thread_id` (CLI flag or API field)

## Phases (implemented)
### Phase 1–2: Core agent (CRM-less)
- Technical support -> KB answer
- Restricted actions -> escalation + log
- Debug routing in CLI

### Phase 3: RAG KB upgrade
- `kb_docs/` stores markdown KB articles
- `scripts/ingest_kb.py` builds `.vectorstore/` (Chroma + OpenAI embeddings)
- Agent prefers vector retrieval; falls back to mock KB if store doesn't exist

### Phase 4: API + speech-ready adapters
- `api.py` provides `/chat` (reply + intent + escalation fields)
- `speech/adapters.py` contains TTS interface stubs + optional local `pyttsx3`

### Phase 5: Eval harness
- `evals/cases.json` defines routing checks
- `evals/run_evals.py` runs PASS/FAIL regression checks

## Setup (Windows / PowerShell)
### 1) Create and activate a virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Required for GPT-4o classification/answers and for embeddings during KB ingest:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

Optional tuning:
```powershell
$env:LLM_MODEL="gpt-4o"
$env:LLM_TEMPERATURE="0.2"
$env:KB_DOCS_DIR="kb_docs"
$env:VECTORSTORE_DIR=".vectorstore"
$env:RAG_TOP_K="4"
```

## Run (Phase 1–2): CLI
```bash
python main.py --debug
```

Use a different conversation memory thread:
```bash
python main.py --debug --thread-id demo-1
```

## Run (Phase 3): Build the vector KB
This requires `OPENAI_API_KEY` (embeddings):
```bash
python scripts/ingest_kb.py
```

Then ask router/cache questions; the agent will prefer RAG retrieval.

## Run (Phase 4): HTTP API server
Start server:
```bash
uvicorn api:app --reload
```

Example request (PowerShell):
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -ContentType "application/json" -Body (@{
  text = "How do I reset my router?"
  thread_id = "web-1"
  debug = $true
} | ConvertTo-Json)
```

## Run (Phase 4): Optional local TTS
TTS is optional. If you want voice output:
```bash
pip install pyttsx3
python main.py --debug --tts
```

## Run (Phase 5): Evals (regression checks)
```bash
python evals/run_evals.py
```

## Operational notes
### Restricted action logging
- File: `requests_log.json`
- Entries include timestamp + category + user text + reason (+ optional extra payload)

### Debug mode
- CLI: `python main.py --debug`
- API: send `"debug": true`

### If you do NOT set OPENAI_API_KEY
- Classifier uses heuristic fallback.
- Knowledge answers use RAG only if `.vectorstore/` exists; otherwise they fall back to mock KB.

## Troubleshooting
- If ingest fails: confirm `OPENAI_API_KEY` is set and dependencies installed.
- If RAG returns nothing: confirm `.vectorstore/` exists and you ran `python scripts/ingest_kb.py`.
