"""
Evaluator: answers the "how do you know your RAG system actually works"
question. We use an LLM-as-judge: given the question, the expected answer,
and the system's actual answer, the judge scores semantic correctness on a
0-1 scale and gives one line of reasoning. This is more robust than
exact-match scoring, since a correct answer can be phrased many ways.

Three implementations, same interface, same pattern as datastore.py and
response_generator.py: GeminiJudgeEvaluator (free tier, rate-limited),
OllamaJudgeEvaluator (fully local, no limits), and OpenAIJudgeEvaluator (paid).
"""

import json

from src.interfaces import BaseEvaluator
from src.retry import with_retry

GEMINI_JUDGE_MODEL = "gemini-2.5-flash"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"
OLLAMA_JUDGE_MODEL = "llama3.2"

JUDGE_SYSTEM_PROMPT = """You are grading a Q&A system. Given a question, an expected \
(reference) answer, and the system's actual answer, decide if the actual answer is \
factually consistent with the expected answer. Minor differences in wording or extra \
correct detail are fine. Respond ONLY with a compact JSON object: \
{"score": <0.0 to 1.0>, "reasoning": "<one sentence>"}"""


def _build_user_prompt(question: str, expected: str, actual: str) -> str:
    return f"Question: {question}\nExpected answer: {expected}\nActual answer: {actual}"


def _parse_judgment(raw_text: str) -> dict:
    try:
        result = json.loads(raw_text)
        return {"score": float(result.get("score", 0.0)), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": 0.0, "reasoning": "judge returned unparseable output"}


class GeminiJudgeEvaluator(BaseEvaluator):
    def __init__(self, api_key: str, model: str = GEMINI_JUDGE_MODEL):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def evaluate(self, question: str, expected: str, actual: str) -> dict:
        from google.genai import types

        response = with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=_build_user_prompt(question, expected, actual),
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        )
        return _parse_judgment(response.text)


class OllamaJudgeEvaluator(BaseEvaluator):
    """Fully local judging via Ollama -- no API key, no rate limits."""

    def __init__(self, model: str = OLLAMA_JUDGE_MODEL, host: str | None = None):
        import ollama

        self._client = ollama.Client(host=host) if host else ollama
        self._model = model

    def evaluate(self, question: str, expected: str, actual: str) -> dict:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, expected, actual)},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        return _parse_judgment(response.message.content)


class OpenAIJudgeEvaluator(BaseEvaluator):
    def __init__(self, api_key: str, model: str = OPENAI_JUDGE_MODEL):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def evaluate(self, question: str, expected: str, actual: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, expected, actual)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return _parse_judgment(response.choices[0].message.content)


def build_evaluator(provider: str, api_key: str) -> BaseEvaluator:
    if provider == "gemini":
        return GeminiJudgeEvaluator(api_key=api_key)
    if provider == "openai":
        return OpenAIJudgeEvaluator(api_key=api_key)
    if provider == "ollama":
        return OllamaJudgeEvaluator()
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'gemini', 'openai', or 'ollama')")
