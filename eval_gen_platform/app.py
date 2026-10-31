"""FastAPI backend.  Run with:  uvicorn app:app --reload   then open http://localhost:8000"""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import cache
import ingestion
from chains import get_llm
from config import OPENROUTER_API_KEY
from knowledge import KnowledgeBase
from logger import get_logger
from memory import AgentMemory
from workflow import build_workflow

log = get_logger("app")

# ---------------- startup checks ----------------
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found. Put it in a .env file next to app.py")
if not cache.redis_ok():
    raise RuntimeError("Cannot connect to Redis. Start it first with:  redis-server")

# ---------------- services (created once when the server starts) ----------------
embeddings = cache.get_embeddings()
cache.enable_llm_cache()
kb = KnowledgeBase(embeddings)
answer_cache = cache.AnswerCache(embeddings)

generator_memory = AgentMemory("generator")   # two SEPARATE memories
evaluator_memory = AgentMemory("evaluator")
workflow = build_workflow(
    kb, answer_cache,
    generator_llm=get_llm(temperature=0.2),
    evaluator_llm=get_llm(temperature=0),
    generator_memory=generator_memory,
    evaluator_memory=evaluator_memory,
)

app = FastAPI(title="Evaluator-Generator Platform")


# ---------------- request bodies ----------------
class Question(BaseModel):
    question: str = Field(max_length=1000)


class UrlBody(BaseModel):
    url: str


class WikiBody(BaseModel):
    page: str


# ---------------- helpers ----------------
def run_ingestion(action, *args):
    """Run one ingestion function and turn errors into clear HTTP errors."""
    try:
        return {"message": action(kb, *args)}
    except ValueError as e:                      # validation problems (bad type, empty text, bad URL...)
        raise HTTPException(400, str(e))
    except Exception as e:
        log.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Could not process the source: {e}")


# ---------------- pages and status ----------------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
def health():
    return {"redis": cache.redis_ok(), "documents": len(cache.list_documents())}


# ---------------- knowledge sources ----------------
@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        try:
            message = ingestion.ingest_file(kb, f.filename or "file", f.file.read())
            results.append({"name": f.filename, "ok": True, "message": message})
        except ValueError as e:
            results.append({"name": f.filename, "ok": False, "message": str(e)})
        except Exception as e:
            log.error(f"Ingestion failed for {f.filename}: {e}")
            results.append({"name": f.filename, "ok": False, "message": f"Could not process the file: {e}"})
    return {"results": results}


@app.post("/add-url")
def add_url(body: UrlBody):
    return run_ingestion(ingestion.ingest_url, body.url.strip())


@app.post("/add-wikipedia")
def add_wikipedia(body: WikiBody):
    return run_ingestion(ingestion.ingest_wikipedia, body.page)


@app.get("/documents")
def documents():
    return {"documents": cache.list_documents()}


@app.delete("/knowledge")
def clear_knowledge():
    kb.clear()
    return {"message": "Knowledge base cleared."}


# ---------------- question answering ----------------
@app.post("/ask")
def ask(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    try:
        return workflow.invoke(question)
    except Exception as e:
        log.error(f"Workflow failed: {e}")
        raise HTTPException(500, f"Something went wrong: {e}")