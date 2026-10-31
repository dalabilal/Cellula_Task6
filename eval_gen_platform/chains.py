
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from config import LLM_MODEL, NOT_AVAILABLE, OPENROUTER_API_KEY


def get_llm(temperature: float):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )



GENERATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are the Generator. Answer the user's question using ONLY the context below.

Rules:
- Do not invent anything that the context does not support.
- If the context does not contain the answer, reply exactly: "{not_available}"
- If evaluator feedback is given, fix every point in it.

Your memory (recent history):
{history}

Context:
{context}

Question: {question}

Previous answer: {previous_answer}
Evaluator feedback: {feedback}

Answer:"""
).partial(not_available=NOT_AVAILABLE)


def build_generator_chain(llm, memory):
    prepare = RunnableLambda(
        lambda s: {
            "question": s["question"],
            "context": s["context"],
            "previous_answer": s["answer"] or "none",
            "feedback": s["feedback"] or "none",
            "history": memory.recent("generator"),  
        }
    )
    return prepare | GENERATOR_PROMPT | llm | StrOutputParser()


class Evaluation(BaseModel):
    approved: bool = Field(description="true only if the answer is good enough to return to the user")
    score: int = Field(description="overall quality from 1 to 10")
    feedback: str = Field(description="what must be improved (empty if approved)")


evaluation_parser = PydanticOutputParser(pydantic_object=Evaluation)

EVALUATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are the Evaluator. Judge the answer below.

Criteria:
- Accuracy
- Relevance to the question
- Completeness
- Consistency with the context
- Grounding in the context
- No unsupported claims
- Overall quality

Rule: if the answer says the information is not available AND the context really
does not contain it, that is a correct answer -> approve it.

Your memory (previous evaluations):
{history}

Question: {question}

Context:
{context}

Answer to evaluate:
{answer}

If it needs improvement, give clear and specific feedback.
{format_instructions}"""
).partial(format_instructions=evaluation_parser.get_format_instructions())


def build_evaluator_chain(llm, memory):
    prepare = RunnableLambda(
        lambda s: {
            "question": s["question"],
            "context": s["context"],
            "answer": s["answer"],
            "history": memory.recent("evaluator"),  
        }
    )
    chain = prepare | EVALUATOR_PROMPT | llm | evaluation_parser

    safe_default = RunnableLambda(
        lambda _: Evaluation(approved=False, score=0, feedback="Evaluation failed. Re-check that every claim is in the context.")
    )
    return chain.with_fallbacks([safe_default], exceptions_to_handle=(OutputParserException,))
