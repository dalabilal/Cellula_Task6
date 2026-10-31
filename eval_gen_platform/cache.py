import hashlib

import redis
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_community.storage import RedisStore
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache

from config import CACHE_TTL, EMBEDDING_MODEL, REDIS_URL, SEMANTIC_DISTANCE
from logger import get_logger

log = get_logger("cache")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def redis_ok() -> bool:
    try:
        return r.ping()
    except redis.RedisError:
        return False


# ---------- knowledge version (used to avoid stale cache) ----------
def knowledge_version() -> str:
    return r.get("kb:version") or "0"


def bump_version():
    r.incr("kb:version")


def get_embeddings():
    """HuggingFace embeddings whose results are saved in Redis."""
    base = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = RedisStore(redis_url=REDIS_URL, ttl=CACHE_TTL, namespace="emb")
    return CacheBackedEmbeddings.from_bytes_store(base, store, namespace=EMBEDDING_MODEL)


def enable_llm_cache():
    set_llm_cache(RedisCache(redis_url=REDIS_URL, ttl=CACHE_TTL))



class AnswerCache:
    """Similar questions (same meaning) return the same validated answer."""

    def __init__(self, embeddings):
        self.cache = RedisSemanticCache(
            embeddings=embeddings,
            redis_url=REDIS_URL,
            distance_threshold=SEMANTIC_DISTANCE,
            ttl=CACHE_TTL,
            name="answer_cache",
        )

    def get(self, question: str):
        hits = self.cache.lookup(question, knowledge_version())
        if hits:
            log.info("Semantic cache HIT")
            return hits[0].text
        log.info("Semantic cache MISS")
        return None

    def put(self, question: str, answer: str):
        self.cache.update(question, knowledge_version(), [Generation(text=answer)])



def get_retrieval(query: str):
    return r.get(f"ret:{knowledge_version()}:{_hash(query)}")


def set_retrieval(query: str, context: str):
    r.set(f"ret:{knowledge_version()}:{_hash(query)}", context, ex=CACHE_TTL)



def is_ingested(doc_id: str) -> bool:
    return bool(r.exists(f"doc:{doc_id}"))


def mark_ingested(doc_id: str, name: str, chunks: int):
    r.hset(f"doc:{doc_id}", mapping={"name": name, "chunks": chunks})
    bump_version()


def list_documents() -> list[dict]:
    return [r.hgetall(k) for k in r.scan_iter("doc:*")]


def clear_documents():
    for key in r.scan_iter("doc:*"):
        r.delete(key)
    bump_version()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
