from langchain_chroma import Chroma

import cache
from config import CHROMA_DIR, TOP_K
from logger import get_logger

log = get_logger("knowledge")


class KnowledgeBase:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._connect()

    def _connect(self):
        self.store = Chroma(
            collection_name="knowledge",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

    def add(self, chunks):
        self.store.add_documents(chunks)

    def is_empty(self) -> bool:
        return self.store._collection.count() == 0

    def clear(self):
        self.store.delete_collection()
        self._connect()
        cache.clear_documents()

    def retrieve(self, query: str) -> str:
        """Return the top chunks as one text block (empty string if nothing stored)."""
        if self.is_empty():
            return ""
        cached = cache.get_retrieval(query)
        if cached is not None:
            log.info("Retrieval cache HIT")
            return cached

        docs = self.store.similarity_search(query, k=TOP_K)
        context = "\n\n---\n\n".join(
            f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )
        cache.set_retrieval(query, context)
        return context
