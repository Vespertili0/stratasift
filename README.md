# StrataSift

[![pytest](https://github.com/Vespertili0/stratasift/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/Vespertili0/stratasift/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/Vespertili0/stratasift/branch/main/graph/badge.svg)](https://codecov.io/gh/Vespertili0/stratasift)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

StrataSift is a multi-agent text processing and synthesis pipeline built on **LangGraph**. It implements structural markdown parsing, cyclic fact-checking routing, and semantic indexing powered by a local, offline **LanceDB** vector store to capture and save atomic insights directly into an **Obsidian** vault.

---

## Key Features

1. **Structured Parser**: Extract and segment chemical processing, synthesis methodologies, and experimental results from raw literature.
2. **Hierarchical Synthesis (LangGraph)**:
   * **Supervisor Triage**: Assesses literature relevance against user interests.
   * **Consolidated Specialist**: Performs holistic extraction of methodology, discussion, and parameters in a single pass.
   * **Reflection Agent**: Conducts cyclic fact-checking to eliminate hallucinations.
   * **Supervisor Network**: Queries LanceDB to append or link insights.
3. **Vault-Centric Database**: LanceDB database tables and vector files are housed directly in your Obsidian vault inside the `.stratasift/` directory.
4. **Decoupled Security**: API credentials are isolated from vault configuration settings to prevent accidental leaks.

---

## Vault-Centric Directory Layout

By default, StrataSift works relative to your active Obsidian vault. Set up a `.stratasift` folder inside your vault root:

```
my-scientific-vault/
├── .stratasift/
│   ├── config.json          # Prompts, model mappings, and thresholds
│   ├── lancedb/             # Automatically generated local vector database files
│   └── eval_benchmarks/     # Serialised high-scoring evaluation golden records
├── Paper1.md      # Synthesised notes
└── Paper2.md
```

---

## Initialisation and Configuration

To configure StrataSift, create a `config.json` inside your vault's `.stratasift/` directory. This file is recursively merged with the base `config.yaml` on startup. If any fields are omitted, StrataSift falls back to the defaults.

### Example `.stratasift/config.json`

```json
{
  "system": {
    "domain_interests": [
      "reactions and molecular dynamics at heterogeneous interfaces for energy storage (batteries)",
      "catalytic conversion of furfural and CO2 hydrogenation"
    ],
    "methodology_interests": [
      "AIMD and classical Molecular Dynamics (LAMMPS, GROMACS, OpenMM)",
      "Machine-Learned Interatomic Potentials (MACE, GAP, NNP, Allegro)"
    ]
  },
  "prompts": {
    "triage": "You are the Principal Investigator (Supervisor Agent) directing a team of scientific specialists...",
    "specialist_consolidated": "You are a scientific Specialist Agent performing holistic analysis. Extract key data points based on reading directive: '{directive}'",
    "reflection": "You are a Senior Researcher acting as the Reflection and Fact-Checking Agent...",
    "synthesis": "You are a Senior Researcher. Synthesise the verified scientific facts into an AtomicInsight..."
  }
}
```

---

## API Keys Isolation

To ensure API keys are never committed or shared, you can configure them in one of two ways:

### 1. Environment Variables (Default)
Set the keys directly in your environment:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OLLAMA_CLOUD_API_KEY="your-ollama-api-key"
```

### 2. Credentials File
Set the path to a credentials file under `api_key_file` in your `.stratasift/config.json`. The file should follow a standard environment variable formatting:
```env
# ~/.config/stratasift/credentials.env
GEMINI_API_KEY=your-gemini-api-key
OLLAMA_CLOUD_API_KEY=your-ollama-api-key
```

---

## LLMOps Evaluation & Golden Datasets

StrataSift implements an offline, milestone-driven auditing suite powered by `deepeval`. It computes referenceless metrics (faithfulness, relevance, and recall) over recently generated notes:
* **Faithfulness**: Validates specialist extraction accuracy against original literature segments.
* **Relevance**: Evaluates final synthesized outputs for domain alignment.

Pairs exceeding the configured `threshold` (under the `evaluation` block in `config.yaml`) are automatically serialized into `.stratasift/eval_benchmarks/golden_set.json` to bootstrap a local "Golden Dataset" for future model fine-tuning and regression testing.

---

---

## CLI Usage

Run diagnostic checks, process raw literature files, and execute evaluations using the CLI commands:

```bash
# Verify configurations, paths, and model connectivity
uv run stratasift check

# Scan and ingest raw markdown clips from a directory into the Obsidian vault
uv run stratasift ingest /path/to/raw/clips

# Run referenceless LLM evaluations and bootstrap golden datasets
uv run stratasift eval
```
