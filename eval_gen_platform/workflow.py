import time

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
)

from chains import build_evaluator_chain, build_generator_chain
from config import DISCLAIMER, MAX_LOOPS, NOT_AVAILABLE
from logger import get_logger

log = get_logger("workflow")


def build_workflow(kb, answer_cache, generator_llm, evaluator_llm, generator_memory, evaluator_memory):
    generator_chain = build_generator_chain(generator_llm, generator_memory)
    evaluator_chain = build_evaluator_chain(evaluator_llm, evaluator_memory)


    def start_state(question: str) -> dict:
        return {
            "question": question,
            "cached": answer_cache.get(question),
            "context": "",
            "answer": "",
            "feedback": "",
            "loop": 0,
            "approved": False,
            "attempts": [],
        }

    def retrieve(state: dict) -> str:
        # after the first loop, the Evaluator's feedback also guides retrieval
        return kb.retrieve(f"{state['question']} {state['feedback']}".strip())

    def update_state(state: dict) -> dict:
        verdict = state["verdict"]
        loop = state["loop"]
        log.info(f"Loop {loop}: approved={verdict.approved} score={verdict.score}")

        # each agent writes ONLY to its own memory
        generator_memory.add("generator", f"Q: {state['question']} | Attempt {loop}: {state['answer'][:300]}")
        generator_memory.add("generator", f"Context used (loop {loop}): {state['context'][:200]}")
        evaluator_memory.add(
            "evaluator",
            f"Attempt {loop}: approved={verdict.approved}, score={verdict.score}, feedback={verdict.feedback}",
        )

        attempt = {"loop": loop, "answer": state["answer"], "approved": verdict.approved,
                   "score": verdict.score, "feedback": verdict.feedback}
        return {**state, "approved": verdict.approved, "feedback": verdict.feedback,
                "attempts": state["attempts"] + [attempt]}

    def finalize(state: dict) -> dict:
        answer = state["answer"]
        if not state["approved"]:
            answer += "\n\n" + DISCLAIMER  # max loops reached without approval
        return {"answer": answer, "validated": state["approved"], "loops": state["loop"],
                "attempts": state["attempts"], "from_cache": False}

    def no_knowledge(state: dict) -> dict:
        return {"answer": NOT_AVAILABLE, "validated": True, "loops": 0, "attempts": [], "from_cache": False}

    def from_cache(state: dict) -> dict:
        return {"answer": state["cached"], "validated": True, "loops": 0, "attempts": [], "from_cache": True}

    def save_to_cache(result: dict) -> dict:
        if result["validated"] and result["loops"] > 0:  # only cache answers the Evaluator approved
            answer_cache.put(result["question"], result["answer"])
        return result

    # ---------- ONE iteration: Generator -> Evaluator ----------
    iteration = (
        RunnablePassthrough.assign(loop=lambda s: s["loop"] + 1)
        | RunnablePassthrough.assign(context=RunnableLambda(retrieve))
        | RunnablePassthrough.assign(answer=generator_chain)
        | RunnablePassthrough.assign(verdict=evaluator_chain)
        | RunnableLambda(update_state)
    )

    # ---------- one loop step: stop (pass through) if approved, otherwise iterate ----------
    loop_step = RunnableBranch(
        (lambda s: s["approved"], RunnablePassthrough()),
        iteration,
    )

    # exactly MAX_LOOPS steps = strict maximum of 4 iterations
    feedback_loop = RunnableSequence(*[loop_step] * MAX_LOOPS)

    # ---------- answer pipeline: first retrieval, then the loop ----------
    answer_pipeline = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve))
        | RunnableBranch(
            (lambda s: not s["context"], RunnableLambda(no_knowledge)),  # empty knowledge base
            feedback_loop | RunnableLambda(finalize),
        )
    )

    # ---------- full workflow ----------
    full = (
        RunnableLambda(start_state)
        | RunnableBranch(
            (lambda s: s["cached"] is not None, RunnableLambda(from_cache)),  # semantic cache hit
            answer_pipeline,
        )
    )

    def run(question: str) -> dict:
        start = time.time()
        result = full.invoke(question)
        result["question"] = question
        result = save_to_cache(result)
        log.info(f"Done in {time.time() - start:.1f}s | loops={result['loops']} | "
                 f"validated={result['validated']} | cache={result['from_cache']}")
        return result

    return RunnableLambda(run)
