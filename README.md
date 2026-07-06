# SEC 10-K RAG Assistant

A Retrieval Augmented Generation (RAG) pipeline that answers questions about companies by retrieving relevant passages from their SEC 10-K annual reports and generating grounded answers. Runs entirely locally via Ollama by default (no API key, no rate limits, no cost) -- Gemini's free tier and OpenAI are both supported as drop-in alternatives by changing one setting.

Built as a learning/portfolio project to demonstrate the core RAG pattern used across enterprise AI deployments: chunk documents → embed → store in a vector database → retrieve relevant context at query time → generate a grounded answer → evaluate whether the system actually got it right.

## Problem it solves

Reading a 10-K to answer one specific question (e.g. "how many employees does Apple have?") means searching through 50-150 pages of dense text. This tool lets you index a set of filings once and then ask natural-language questions across all of them, with the system citing which company's filing each answer came from -- and explicitly saying "not found" rather than guessing when a question is out of scope (see the "capital of France" case in `eval/questions.json`).

## Architecture

```
Documents (data/source/*.txt)
     │
     ▼
  Indexer (src/indexer.py)       ──►  splits text into ~400-token, paragraph-aware,
     │                                overlapping chunks
     ▼
 Datastore (src/datastore.py)    ──►  embeds chunks (Gemini or OpenAI), stores
     │                                vectors in a local ChromaDB collection
     ▼
 Retriever (src/retriever.py)    ──►  embeds the query, does similarity search,
     │                                drops results below a minimum similarity threshold
     ▼
ResponseGenerator                ──►  sends query + retrieved chunks to an LLM
(src/response_generator.py)           (Gemini or OpenAI), returns a grounded, cited
     │                                answer -- or a fixed "not found" response if
     │                                retrieval was empty
     ▼
 Evaluator (src/evaluator.py)    ──►  LLM-as-judge: scores each answer against a
                                       reference answer, 0.0-1.0, with reasoning
```

Every component sits behind an abstract interface (`src/interfaces.py`). `src/datastore.py`, `src/response_generator.py`, and `src/evaluator.py` each ship *three* concrete implementations against that interface -- Ollama, Gemini, and OpenAI -- selected at runtime by a single `LLM_PROVIDER` setting in `.env`. That's the payoff of the interface pattern: switching providers is a config change, not a rewrite.

## Corpus

Three real SEC 10-K filings (Business + Risk Factors sections, pulled directly from SEC EDGAR), fiscal year 2024:
- Apple Inc.
- Tesla, Inc.
- Microsoft Corporation

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# default LLM_PROVIDER=ollama needs no key -- see Ollama setup below
```

### Option A: Ollama (default -- fully local, no key, no limits)

1. Install Ollama from [ollama.com](https://ollama.com) (Mac/Windows/Linux app).
2. Pull the two models this project uses:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Make sure the Ollama app/server is running (it starts automatically after install, or run `ollama serve`), then just run the pipeline -- `LLM_PROVIDER=ollama` is already the default in `.env.example`.

Everything runs on your machine: no API key, no per-request rate limit, no cost. The tradeoff is answer quality is a step below Gemini/GPT-4o-mini (llama3.2 is a small 3B model), and speed depends on your hardware -- a modern laptop handles this project's scale fine, but each query will feel slower than a hosted API.

### Option B: Gemini (free tier, no credit card, but rate-limited)

Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`. Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) -- sign in with a Google account, click "Create API key." As of mid-2026 the free tier can be as tight as 5 requests/minute on `gemini-2.5-flash` depending on your account, which `evaluate`/`run` can hit since they fire off many calls back to back -- the Gemini code paths already retry with exponential backoff on rate-limit errors, so a run that hits the limit will pause and continue rather than crash, just slower.

### Option C: OpenAI (paid, ~$0.10 for this whole project)

Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`. Get a key at [platform.openai.com](https://platform.openai.com) (requires adding billing). Fastest and most capable of the three options, no free tier.

## Usage

```bash
# Index the documents (embeds + stores in ./chroma_db)
python main.py add -p data/source/

# Ask a question
python main.py query "What is Tesla's mission?"

# Clear the vector store
python main.py reset

# Run the evaluation set (11 questions, including one deliberately out-of-scope)
python main.py evaluate -f eval/questions.json

# Do all three in one shot: reset, index, evaluate
python main.py run
```

## Example

```
$ python main.py query "What are Microsoft's three reportable segments?"

Q: What are Microsoft's three reportable segments?
A: According to Microsoft's 10-K filing, the company reports its financial
performance using three segments: Productivity and Business Processes,
Intelligent Cloud, and More Personal Computing.

Sources:
  - microsoft_10k_2024.txt (similarity=0.612)
```

## Evaluation

`eval/questions.json` has 11 question/expected-answer pairs spanning all three filings, plus one deliberately out-of-scope question ("What is the capital of France?") to verify the system correctly refuses to answer instead of hallucinating from general knowledge. `python main.py evaluate` runs each question through the full pipeline and uses an LLM-as-judge (same provider as everything else) to score the actual answer against the reference, printing a per-question score and an overall average.
