"""
ResponseGenerator: builds a grounded prompt from retrieved chunks and calls
an LLM. If retrieval came back empty, we don't call the LLM at all -- we
return a fixed "not found" answer. This is the guard against hallucination:
a RAG system that always tries to answer, even with no relevant context, is
one that will confidently make things up.

Three implementations, all against the same BaseResponseGenerator interface:
GeminiResponseGenerator (free tier, rate-limited), OllamaResponseGenerator
(fully local, no limits), and OpenAIResponseGenerator (paid). See
src/datastore.py for the same pattern applied to embeddings.
"""

from src.interfaces import BaseResponseGenerator, RagAnswer, RetrievedChunk
from src.retry import with_retry

GEMINI_CHAT_MODEL = "gemini-flash-latest"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OLLAMA_CHAT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's question \
using ONLY the provided context excerpts from SEC 10-K filings. \
Cite which company's filing each piece of information comes from. \
If the context does not contain enough information to answer, say so explicitly \
instead of guessing."""

NOT_FOUND_MESSAGE = (
    "I couldn't find anything in the indexed documents relevant to that question. "
    "Try rephrasing, or check that the right filings are indexed."
)


def _build_context_block(context: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[Source: {rc.chunk.source}]\n{rc.chunk.text}" for rc in context)


class GeminiResponseGenerator(BaseResponseGenerator):
    def __init__(self, api_key: str, model: str = GEMINI_CHAT_MODEL):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[RetrievedChunk]) -> RagAnswer:
        if not context:
            return RagAnswer(question=question, answer=NOT_FOUND_MESSAGE, sources=[], grounded=False)

        from google.genai import types

        user_prompt = f"Context:\n{_build_context_block(context)}\n\nQuestion: {question}"
        response = with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
        )
        return RagAnswer(question=question, answer=response.text, sources=context, grounded=True)


class OllamaResponseGenerator(BaseResponseGenerator):
    """Fully local generation via Ollama -- no API key, no rate limits, no
    cost. Requires Ollama installed (https://ollama.com) and the chat model
    pulled once, e.g. `ollama pull llama3.2`."""

    def __init__(self, model: str = OLLAMA_CHAT_MODEL, host: str | None = None):
        import ollama

        self._client = ollama.Client(host=host) if host else ollama
        self._model = model

    def generate(self, question: str, context: list[RetrievedChunk]) -> RagAnswer:
        if not context:
            return RagAnswer(question=question, answer=NOT_FOUND_MESSAGE, sources=[], grounded=False)

        user_prompt = f"Context:\n{_build_context_block(context)}\n\nQuestion: {question}"
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        return RagAnswer(question=question, answer=response.message.content, sources=context, grounded=True)


class OpenAIResponseGenerator(BaseResponseGenerator):
    def __init__(self, api_key: str, model: str = OPENAI_CHAT_MODEL):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[RetrievedChunk]) -> RagAnswer:
        if not context:
            return RagAnswer(question=question, answer=NOT_FOUND_MESSAGE, sources=[], grounded=False)

        user_prompt = f"Context:\n{_build_context_block(context)}\n\nQuestion: {question}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        answer_text = response.choices[0].message.content
        return RagAnswer(question=question, answer=answer_text, sources=context, grounded=True)


def build_response_generator(provider: str, api_key: str) -> BaseResponseGenerator:
    if provider == "gemini":
        return GeminiResponseGenerator(api_key=api_key)
    if provider == "openai":
        return OpenAIResponseGenerator(api_key=api_key)
    if provider == "ollama":
        return OllamaResponseGenerator()
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'gemini', 'openai', or 'ollama')")
