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

Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env`. Get a free key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) -- sign in with a Google account, click "Create API key." As of mid-2026 the free tier can be as tight as 5 requests/minute depending on your account, which `evaluate`/`run` can hit since they fire off many calls back to back. This project uses `gemini-flash-latest`, an alias Google keeps pointed at whatever Flash model is current, so it won't break again the way pinning a specific version number (e.g. `gemini-2.5-flash`) eventually will as older models get retired -- the Gemini code paths already retry with exponential backoff on rate-limit errors, so a run that hits the limit will pause and continue rather than crash, just slower.

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

## Design decisions

- **Paragraph-aware chunking with overlap**: chunks are built from whole paragraphs (never split mid-sentence) up to ~400 tokens, with 60 tokens of overlap carried into the next chunk so an answer that straddles a chunk boundary is still retrievable.
- **Similarity threshold in the retriever, not the datastore**: `SimpleRetriever` filters out low-similarity matches before they ever reach the LLM. This is what stops the system from confidently answering questions the corpus doesn't cover.
- **No-context guard in the generator**: if retrieval returns nothing, the generator skips the LLM call entirely and returns a fixed "not found" message -- cheaper and safer than letting the model try to answer anyway.
- **Provider swap via config, not code**: `build_embedding_function`, `build_response_generator`, and `build_evaluator` are small factory functions that return an Ollama, Gemini, or OpenAI implementation of the same interface based on `LLM_PROVIDER`, so switching providers is a config change rather than a rewrite -- the same pattern used in production systems to avoid vendor lock-in.
- **Retry with backoff on rate limits**: `src/retry.py` wraps the Gemini API calls so a 429 (rate limit exceeded) triggers a wait-and-retry instead of crashing the whole `evaluate` run, since Gemini's free tier can be rate-limited as low as 5 requests/minute.
- **tiktoken with a fallback**: exact token counts via `tiktoken`, but if its encoding file can't be downloaded (e.g. restricted network), chunking falls back to a characters-per-token estimate rather than failing.

## Web UI

`app.py` wraps the same pipeline (`src/indexer.py`, `src/datastore.py`, `src/retriever.py`, `src/response_generator.py` -- completely unchanged) in a Streamlit front-end: a text box, example-question buttons, and an expandable sources panel showing which chunks and similarity scores backed each answer.

Run it locally:
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Works with any provider set in `.env`, including Ollama.

### Deploying it publicly (free)

1. Push this repo to GitHub (see the GitHub section below if you haven't already).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click "New app," pick this repo and branch, and set the main file to `app.py`.
3. In the app's Settings → Secrets, add (in TOML format):
   ```toml
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "your_actual_key"
   ```
   **Use Gemini or OpenAI here, not Ollama** -- Streamlit Community Cloud runs your app on Google's servers, which have no way to reach a locally-running Ollama instance on your own machine. Ollama is for `streamlit run app.py` on your laptop only.
4. Deploy. You'll get a public URL like `https://your-app-name.streamlit.app`.

Point your portfolio site at that URL directly, or embed it inline with:
```html
<iframe src="https://your-app-name.streamlit.app" width="100%" height="800"></iframe>
```
so it opens right there on the page instead of sending visitors away to GitHub.

### Deploying on AWS (ECS Express Mode)

This repo also deploys on actual AWS infrastructure via [Amazon ECS Express Mode](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-mode.html), which takes a container image and provisions a Fargate service, load balancer, auto-scaling, and security groups behind a single command -- similar in spirit to App Runner, but it deploys a pre-built image instead of building from source. (App Runner itself stopped accepting new customers as of April 30, 2026, so it isn't an option for a new AWS account -- ECS Express Mode is AWS's own recommended replacement.)

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed locally, to build the image
- [AWS CLI](https://aws.amazon.com/cli/) installed and configured (`aws configure`) with an IAM user that has ECS/ECR/IAM permissions
- An AWS account (create one at [aws.amazon.com](https://aws.amazon.com) if you don't have one)

**1. Build the image and push it to Amazon ECR**

```bash
# Build the image (uses the Dockerfile in this repo)
docker build -t filingiq .

# Create an ECR repository (one-time)
aws ecr create-repository --repository-name filingiq --region us-east-1

# Authenticate Docker to ECR, then tag and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag filingiq:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/filingiq:latest
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/filingiq:latest
```

**2. Create the two IAM roles ECS Express Mode requires**

`ecsTaskExecutionRole` (lets ECS pull the image and write logs) and `ecsInfrastructureRoleForExpressServices` (lets ECS create the load balancer and related networking on your behalf). The easiest way to create both correctly is AWS's own guided setup -- see [Setting up ECS Express Mode](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-mode-setting-up.html), which creates the roles with the exact trust policies Express Mode expects.

**3. Create the service**

```bash
aws ecs create-express-gateway-service \
  --execution-role-arn arn:aws:iam::<your-account-id>:role/ecsTaskExecutionRole \
  --infrastructure-role-arn arn:aws:iam::<your-account-id>:role/ecsInfrastructureRoleForExpressServices \
  --primary-container '{
    "image": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/filingiq:latest",
    "containerPort": 8080,
    "environment": [
      {"name": "LLM_PROVIDER", "value": "gemini"},
      {"name": "GEMINI_API_KEY", "value": "your_actual_key"}
    ]
  }' \
  --service-name "filingiq" \
  --health-check-path "/" \
  --scaling-target '{"minTaskCount":1,"maxTaskCount":2}' \
  --monitor-resources
```

This provisions the Fargate service, an Application Load Balancer, and auto-scaling in one call, and prints a public URL when it's ready.

There's also a console path (Amazon ECS → "Create an Express Mode service") that walks through the same steps with a UI instead of flags, and can create the IAM roles for you as part of the wizard -- worth using for a first deployment if the CLI flags feel like a lot at once.

**Cost note:** ECS Express Mode has no extra charge on top of the underlying resources (Fargate compute, the load balancer, data transfer). Delete the service when you're not actively demoing it to avoid ongoing charges.

## Future work

- A tool-using agent layer on top of the same retrieval pipeline, so the system can decide when to search the corpus versus answer directly.
- Broader corpus coverage (full filings rather than the Business + Risk Factors sections currently indexed).
- A hosted vector store for larger corpora, if the local ChromaDB persistence stops being sufficient.
