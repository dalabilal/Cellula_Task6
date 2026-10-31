# Evaluator–Generator Q&A Platform

A question-answering app where a **Generator LLM** answers from your documents and an **Evaluator LLM**
checks the answer and sends feedback until it is good (max **4 loops**).

## Architecture

```
User question
   │
   ▼
Semantic cache (Redis) ── hit ──► cached validated answer
   │ miss
   ▼
Retrieve context (Chroma)  ── empty KB ──► "information not available"
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

Ingestion: `Load → Extract → Chunk → Embed → Store (Chroma)`; audio goes through Whisper speech-to-text first.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (uploads, URLs, chat) |
| `workflow.py` | LCEL orchestration + loop counter |
| `chains.py` | Generator chain and Evaluator chain |
| `memory.py` | Isolated memory per agent |
| `ingestion.py` | Loaders for all formats + chunking |
| `loader.py` | WAV speech-to-text (faster-whisper) |
| `knowledge.py` | Chroma vector store + retrieval |
| `cache.py` | All Redis caching |
| `config.py`, `logger.py` | Settings and logging |

## LCEL building blocks used

- **RunnablePassthrough.assign** – adds `loop`, `context`, `answer`, `verdict` to the state.
- **RunnableLambda** – wraps normal Python functions (retrieve, update state, finalize).
- **RunnableBranch** – cache hit?, empty knowledge base?, already approved?
- **RunnableSequence** – chains exactly `MAX_LOOPS` (4) loop steps, which makes the limit strict.

## Redis caching

| What | How |
|---|---|
| Embeddings | `CacheBackedEmbeddings` + `RedisStore` |
| Generator / Evaluator LLM responses | `RedisCache` (exact prompt match) |
| Final validated answers | `RedisSemanticCache` (similar questions share an answer) |
| Retrieval results | key = knowledge version + query |
| Processed documents | key = file hash, so the same file is not re-ingested |

**No stale data:** every ingestion or clear raises a *knowledge version* number. Cached answers and
retrieval results are tied to it, so after the knowledge changes, old entries never match. Everything also expires after 24 h.
Only answers approved by the Evaluator go into the semantic cache.

## Setup (WSL Ubuntu)

```bash
# 1. Redis Stack (needs Docker Desktop with WSL integration)
docker compose up -d

# 2. Python environment
sudo apt update && sudo apt install -y python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. API key
cp .env.example .env      # then put your OPENROUTER_API_KEY in .env

# 4. Run
streamlit run app.py      # open http://localhost:8501 in your Windows browser
```

Optional: `sudo apt install -y libreoffice-impress` to support old `.ppt` files (`.pptx` works without it).

## Error handling and logging

- File type/size, URL format, empty text, Redis down, missing API key are all checked with clear messages.
- Bad Evaluator output counts as "not approved" instead of crashing.
- Logs go to the terminal and `logs/app.log` (loop number, score, cache hits/misses, time per question).
