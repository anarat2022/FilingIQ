"""
CLI entrypoint for the RAG pipeline.

Provider defaults to Gemini (free tier, no credit card required). Set
LLM_PROVIDER=openai in .env to switch to OpenAI instead -- nothing else in
the codebase needs to change, which is the whole point of the interfaces in
src/interfaces.py.

Usage:
    python main.py add -p data/source/            # index documents
    python main.py query "What is Tesla's mission?"
    python main.py reset                          # clear the vector store
    python main.py evaluate -f eval/questions.json
    python main.py run                            # reset + add + evaluate, all in one
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from src.datastore import ChromaDatastore, build_embedding_function
from src.evaluator import build_evaluator
from src.indexer import SimpleTextIndexer
from src.response_generator import build_response_generator
from src.retriever import SimpleRetriever

load_dotenv()

DEFAULT_SOURCE_PATH = "data/source/"
DEFAULT_EVAL_PATH = "eval/questions.json"


def _get_provider_and_key() -> tuple[str, str]:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "ollama":
        return provider, ""  # no API key needed -- talks to a local Ollama server

    env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        print(f"ERROR: {env_var} is not set. Copy .env.example to .env and fill it in.")
        print(f"(Provider is '{provider}' -- set LLM_PROVIDER in .env to 'gemini', 'openai', or 'ollama'.)")
        sys.exit(1)
    return provider, key


def build_pipeline():
    provider, api_key = _get_provider_and_key()
    indexer = SimpleTextIndexer()
    embedding_fn = build_embedding_function(provider, api_key)
    datastore = ChromaDatastore(embedding_function=embedding_fn)
    retriever = SimpleRetriever(datastore)
    generator = build_response_generator(provider, api_key)
    return indexer, datastore, retriever, generator


def cmd_add(path: str):
    indexer, datastore, _, _ = build_pipeline()
    chunks = indexer.load_and_chunk(path)
    stored = datastore.add(chunks)
    print(f"Indexed {stored} chunks from '{path}'. Datastore now has {datastore.count()} chunks total.")


def cmd_reset():
    _, datastore, _, _ = build_pipeline()
    datastore.reset()
    print("Datastore cleared.")


def cmd_query(question: str):
    _, _, retriever, generator = build_pipeline()
    context = retriever.retrieve(question, top_k=5)
    result = generator.generate(question, context)

    print(f"\nQ: {result.question}")
    print(f"A: {result.answer}\n")
    if result.sources:
        print("Sources:")
        for rc in result.sources:
            print(f"  - {rc.chunk.source} (similarity={rc.score:.3f})")
    else:
        print("Sources: none (question was not grounded in the corpus)")


def cmd_evaluate(eval_path: str):
    provider, api_key = _get_provider_and_key()
    _, _, retriever, generator = build_pipeline()
    judge = build_evaluator(provider, api_key)

    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total_score = 0.0
    print(f"Running {len(cases)} evaluation questions from '{eval_path}' (provider: {provider})...\n")
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case["expected_answer"]

        context = retriever.retrieve(question, top_k=5)
        result = generator.generate(question, context)
        judgment = judge.evaluate(question, expected, result.answer)

        total_score += judgment["score"]
        print(f"[{i}/{len(cases)}] score={judgment['score']:.2f}  {question}")
        print(f"    reasoning: {judgment['reasoning']}")

    avg = total_score / len(cases) if cases else 0.0
    print(f"\nAverage score: {avg:.2%} across {len(cases)} questions.")


def cmd_run():
    _, datastore, _, _ = build_pipeline()
    datastore.reset()
    cmd_add(DEFAULT_SOURCE_PATH)
    cmd_evaluate(DEFAULT_EVAL_PATH)


def main():
    parser = argparse.ArgumentParser(description="Simple RAG pipeline over SEC 10-K filings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Index documents into the datastore.")
    add_parser.add_argument("-p", "--path", default=DEFAULT_SOURCE_PATH)

    subparsers.add_parser("reset", help="Clear the datastore.")

    query_parser = subparsers.add_parser("query", help="Ask a question.")
    query_parser.add_argument("question", type=str)

    eval_parser = subparsers.add_parser("evaluate", help="Run the evaluation question set.")
    eval_parser.add_argument("-f", "--file", default=DEFAULT_EVAL_PATH)

    subparsers.add_parser("run", help="Reset, index, and evaluate in one shot.")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args.path)
    elif args.command == "reset":
        cmd_reset()
    elif args.command == "query":
        cmd_query(args.question)
    elif args.command == "evaluate":
        cmd_evaluate(args.file)
    elif args.command == "run":
        cmd_run()


if __name__ == "__main__":
    main()
