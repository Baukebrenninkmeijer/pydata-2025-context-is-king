# Context Is King — PyData Amsterdam 2025

Research project comparing long-context LLMs against Retrieval-Augmented Generation (RAG) for data grounding, implementing the Chroma Context Rot methodology.

## Quick start

```bash
# Install
uv sync

# Configure
cp .env.example .env   # set ORQ_API_KEY

# Download source datasets
ck-download-datasets

# Run main experiments
python src/context_is_king/experiments/context_advantage/context_advantage_experiment.py --quick-test
python src/context_is_king/experiments/context_window_scaling/run_experiment.py --quick-test
python src/context_is_king/experiments/reranking_value/run_experiment.py --help
```

## Layout

```
data/                     # Datasets + processed artifacts
  LongMemEval/            # External eval dataset
  arxiv_papers/           # Source docs
  paul_graham_essays/     # Source docs
  processed/              # WikiQA + NQ chunked/filtered outputs
  context_advantage/      # Fixed QAs, needles, haystacks, grouped LongMemEval
  vector_stores/          # ChromaDB persistent stores (gitignored)
results/                  # Experiment outputs
  context_advantage/
  reranking_value/
  scaling/                # scaling/{experiments,analysis}/
  speed_benchmark/
src/context_is_king/      # Package
  models/                 # ModelInterface, ModelConfig
  evaluation/             # Evaluators, metrics
  experiments/            # context_advantage, context_window_scaling, reranking_value, scaling.py
  rag_pipeline/           # Retrieval, reranking, context assembly
  query_generation/       # WikiText query pipeline
  data_ingestion/         # Dataset downloaders, Chroma ingestion
  datasets/               # Dataset-specific utilities (LongMemEval)
  cli/                    # Entrypoints registered in pyproject
notebooks/                # Exploratory notebooks, per experiment
scripts/                  # Standalone scripts + smoke_tests/
blog/                     # Blog post Quarto source
presentation.qmd          # Talk slides (local only)
img/                      # Presentation + blog figures
docs/                     # Quarto render output (gitignored)
```

## Key entrypoints

Installed CLI commands (see `pyproject.toml` `[project.scripts]`):

- `ck-download-datasets` — Fetch Paul Graham, arXiv, Chroma needles
- `ck-ingest-data`, `ck-ingest-chroma` — Ingest into ChromaDB
- `ck-generate-queries-v2` — WikiText query pipeline (v2 is current)
- `ck-extract-longmemeval-ground-truth` — LongMemEval ground-truth extraction

## Guidelines

- LLM calls go through ORQ proxy (use `ORQ_API_KEY`, OpenAI/Azure SDK compatible).
- Default model: `azure/gpt-4.1-mini`. For LLM filtering tasks, avoid `gpt-5-nano` (rate-limited).
- Embeddings: `openai/text-embedding-3-small` (Azure) or `text-embedding-3-small` (OpenAI).
- Python 3.10+ native typing (`list`, `dict`, `x | y`).

## Reproducing the talk

```bash
quarto render presentation.qmd
quarto render blog/index.qmd
```

Output goes to `docs/`.
