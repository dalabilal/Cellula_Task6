# Evaluator–Generator Q&A Platform

A question-answering app where a **Generator LLM** answers only from your documents and an
**Evaluator LLM** checks the answer and sends feedback until it is good (maximum **4 loops**).
Built with LangChain (LCEL), Chroma, Redis, Streamlit and OpenRouter. Runs on **WSL Ubuntu, no Docker**.

## How it works

```
User question
   │
   ▼
Semantic cache (Redis) ── hit ──► cached validated answer
   │ miss
   ▼
Retrieve context (Chroma) ── empty knowledge base ──► "information not available"
   │
   ▼
┌──────────── loop (max 4) ────────────┐
│ Generator LLM  ◄── Generator memory  │
│      │ answer                        │
│      ▼                               │
│ Evaluator LLM  ◄── Evaluator memory  │
│      │                               │
│  approved? ── yes ─► exit            │
│      │ no: feedback → Generator      │
└──────────────────────────────────────┘
   │ 4 loops, still not approved
   ▼
Latest answer + disclaimer
```

Ingestion pipeline: `Load → Extract → Chunk → Embed → Store (Chroma)`.
Audio (WAV) goes through Whisper speech-to-text first.

Supported sources: PDF, DOCX, TXT, source-code files, PPT/PPTX, web pages (URL), Wikipedia pages, WAV audio.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload files, add URLs/Wikipedia, chat, show Generator ↔ Evaluator attempts |
| `workflow.py` | LCEL orchestration: cache check, retrieval, 4-loop Generator ↔ Evaluator process |
| `chains.py` | Generator chain, Evaluator chain, OpenRouter LLM connection |
| `memory.py` | Isolated memory per agent (each memory has an owner) |
| `ingestion.py` | Loaders for every format + chunking |
| `loader.py` | WAV speech-to-text with faster-whisper |
| `knowledge.py` | Chroma vector store and retrieval |
| `cache.py` | All Redis caching |
| `config.py` | Settings (model, chunk size, max loops, Redis URL) |
| `logger.py` | Logging to the terminal and `logs/app.log` |
| `requirements.txt` | Python libraries |
| `.env` / `.env.example` | Your OpenRouter key (`.env` is private, never upload it) |

Folders created when you run the app: `chroma_db/` (stored chunks), `uploads/` (uploaded files), `logs/`.

## LCEL building blocks used (in `workflow.py`)

- **RunnablePassthrough.assign**: adds `loop`, `context`, `answer` and `verdict` to the state.
- **RunnableLambda**: wraps normal Python functions (retrieve, update state, finalize).
- **RunnableBranch**: cache hit? empty knowledge base? answer already approved?
- **RunnableSequence**: chains exactly `MAX_LOOPS` (4) loop steps, so the limit is strict.

## Independent memories

The Generator and the Evaluator each get their own `AgentMemory` object. Every read or write
must say who is calling, and a wrong caller raises `PermissionError`, so neither agent can
access the other's memory.

- Generator memory: conversation questions, previous answers, retrieved context.
- Evaluator memory: evaluation history, decisions, feedback.

## Redis caching

| What | How |
|---|---|
| Embeddings | `CacheBackedEmbeddings` + `RedisStore` |
| Generator / Evaluator LLM responses | `RedisCache` (exact prompt match) |
| Final validated answers | `RedisSemanticCache` (similar questions share an answer) |
| Retrieval results | key = knowledge version + query |
| Processed documents | key = file hash, so the same file is not processed twice |

**No stale data:** every ingestion or clear raises a *knowledge version* number. Cached answers and
retrieval results are tied to that number, so after the knowledge changes, old entries never match.
Entries also expire after 24 hours. Only answers approved by the Evaluator are saved in the semantic cache.

## Setup (WSL Ubuntu, no Docker)

Tested with **Python 3.12**. Some libraries do not support Python 3.14 yet, so use `uv` to get 3.12.
WSL should have about 6 GB of RAM (set it in `%USERPROFILE%\.wslconfig` with `memory=6GB`, then run `wsl --shutdown` in Windows).

**1. Install Redis (once)**
```bash
sudo apt update
sudo apt install -y lsb-release curl gpg
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt update
sudo apt install -y redis
```
Check it: `redis-server` in one terminal, then `redis-cli ping` (should print `PONG`) and `redis-cli MODULE LIST` (should list `search`).

**2. Python environment (once)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 ~/venv312
source ~/venv312/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

**3. API key (once)**
```bash
cp .env.example .env
nano .env        # put your OPENROUTER_API_KEY, then Ctrl+O, Enter, Ctrl+X
```

## Run it

Terminal 1:
```bash
redis-server
```
Terminal 2:
```bash
source ~/venv312/bin/activate
cd /path/to/eval_gen_platform
streamlit run app.py
```
Open `http://localhost:8501` in your Windows browser.

## How to use

1. In the sidebar, add at least one source: upload files and click **Process files**, or add a web page URL, or a Wikipedia page.
2. Ask a question about what is inside your sources.
3. Read the answer: ✅ validated, ⚠️ not fully validated (after 4 loops), or ⚡ served from the Redis semantic cache.
4. Open **Show Generator ↔ Evaluator attempts** to see every loop with its score and feedback.

Questions that your sources cannot answer get: *"The required information is not available in the provided sources."*

Optional: `sudo apt install -y libreoffice-impress` to support old `.ppt` files (`.pptx` works without it).

## Error handling and logging

- Checked with clear messages: unsupported file type, file too large, invalid URL, empty text, Redis not running, missing API key.
- Badly formatted Evaluator output counts as "not approved" instead of crashing.
- `logs/app.log` records loop number, score, cache hits/misses and time per question.

Useful Redis checks: `redis-cli dbsize`, `redis-cli FT._LIST`, `grep -i cache logs/app.log`.
