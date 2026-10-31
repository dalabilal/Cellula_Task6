import streamlit as st

import cache
import ingestion
from chains import get_llm
from config import OPENROUTER_API_KEY
from knowledge import KnowledgeBase
from logger import get_logger
from memory import AgentMemory
from workflow import build_workflow

log = get_logger("app")
st.set_page_config(page_title="Evaluator–Generator Platform", page_icon="🧠", layout="wide")
st.title("Evaluator–Generator Q&A Platform")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key.")
    st.stop()
if not cache.redis_ok():
    st.error("Cannot connect to Redis. Start it with:  docker compose up -d")
    st.stop()


@st.cache_resource
def get_services():
    embeddings = cache.get_embeddings()
    cache.enable_llm_cache()
    return KnowledgeBase(embeddings), cache.AnswerCache(embeddings)


kb, answer_cache = get_services()

if "workflow" not in st.session_state:
    generator_memory = AgentMemory("generator")   
    evaluator_memory = AgentMemory("evaluator")
    st.session_state.workflow = build_workflow(
        kb, answer_cache,
        generator_llm=get_llm(temperature=0.2),
        evaluator_llm=get_llm(temperature=0),
        generator_memory=generator_memory,
        evaluator_memory=evaluator_memory,
    )
    st.session_state.messages = []


def run_ingestion(action, *args):
    """Run an ingestion function and show success/error nicely."""
    try:
        with st.spinner("Processing..."):
            st.sidebar.success(action(kb, *args))
    except Exception as e: 
        log.error(f"Ingestion failed: {e}")
        st.sidebar.error(str(e))



with st.sidebar:
    st.header("Knowledge sources")

    files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, code, PPT/PPTX, WAV)",
        type=[e.lstrip(".") for e in sorted(ingestion.SUPPORTED_EXT)],
        accept_multiple_files=True,
    )
    if st.button("Process files") and files:
        for f in files:
            run_ingestion(ingestion.ingest_file, f.name, f.getvalue())

    url = st.text_input("Web page URL")
    if st.button("Add web page") and url:
        run_ingestion(ingestion.ingest_url, url)

    wiki = st.text_input("Wikipedia page (title or URL)")
    if st.button("Add Wikipedia page") and wiki:
        run_ingestion(ingestion.ingest_wikipedia, wiki)

    st.divider()
    st.subheader("In the knowledge base")
    docs = cache.list_documents()
    for d in docs:
        st.write(f"• {d['name']} ({d['chunks']} chunks)")
    if not docs:
        st.caption("Nothing added yet.")
    if st.button("Clear knowledge base"):
        kb.clear()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your sources...")
if question:
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generator and Evaluator are working..."):
                result = st.session_state.workflow.invoke(question)
            st.markdown(result["answer"])

            if result["from_cache"]:
                st.caption(" Served from the Redis semantic cache")
            elif result["loops"]:
                status = "validated" if result["validated"] else " not fully validated"
                st.caption(f"{status} after {result['loops']} loop(s)")
            with st.expander("Show Generator ↔ Evaluator attempts"):
                for a in result["attempts"]:
                    st.markdown(f"**Loop {a['loop']}** — approved: `{a['approved']}`, score: `{a['score']}`")
                    st.write(a["answer"])
                    if a["feedback"]:
                        st.info(f"Evaluator feedback: {a['feedback']}")
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as e:
            log.error(f"Workflow failed: {e}")
            st.error(f"Something went wrong: {e}")
