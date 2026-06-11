<div align="center">

# AgentConsis-Verify

**Small Language Model – based Multi-Agent System with Self-Consistency and Score Selection**

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Benchmark](https://img.shields.io/badge/Benchmark-GAIA-purple)
![Agents](https://img.shields.io/badge/Multi--Agent-SLM-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

</div>

## Description

AgentConsis-Verify is a local multi-agent reasoning framework for GAIA-style question answering.
It runs multiple small language model agents, asks them to solve the same task
several times, aggregates self-consistency, optionally lets agents use tools
during Stage 1, and then applies cross-agent reasoning-step judging in Stage 2.

The main entry point is:

```bash
python run_gaia.py
```

For a small smoke test:

```bash
python run_gaia.py --level 1 --max-samples 1 --log-name gaia_level1_test
```

For tool-enabled Stage 1 reasoning:

```bash
python run_gaia.py --level 2 --max-samples 1 --enable-stage1-tool-use --log-name gaia_level2_tool_test
```

## News

- Added GAIA batch runner with per-task JSON export and Markdown report export.
- Added Stage 1 parallel multi-agent self-consistency.
- Added optional Stage 1 tool-use trajectory.
- Added Stage 2 pairwise judge scoring over reasoning steps.
- Added rule-based answer validation and penalty calculation.
- Added evidence-oriented search flow with embedding-delta query salience,
  SEER-style source cleaning, full-page fetch, chunk labeling, and retrieval control.
- Decoupled `SearchTool` into a backend adapter; filtering, fetching, and
  evidence construction now live under `search_result_builder`.
- Added search pipeline diagnostics to per-task JSON export.

## Key Features

- **Multi-agent Stage 1 reasoning**: multiple SLM agents answer the same task in parallel.
- **Self-consistency scoring**: repeated answers are normalized and grouped into confidence scores.
- **Optional tool use**: Stage 1 agents can request tools before producing a final answer.
- **Tool cache**: identical tool calls reuse cached results.
- **Cross-agent Stage 2 judging**: agents judge other agents' reasoning steps.
- **Step-level scoring**: judge score is computed over explicit reasoning steps.
- **Rule-based penalties**: malformed answers, tool-call-as-answer, refusal-like answers, and tool failures can reduce score.
- **Evidence-oriented search planning**: generates search queries from
  embedding-delta salient spans and optionally performs next-hop retrieval.
- **Decoupled search adapter**: `SearchTool` only calls search backends and
  returns normalized raw results; evidence construction is handled separately.
- **GAIA-oriented evaluation**: includes dataset loading, attachment preparation, batch running, per-task export, Markdown reporting, and accuracy statistics.

## Overview

AgentConsis-Verify is designed around the following question:

> Can a group of small local agents improve task reliability by reasoning independently, comparing answers, and judging each other's reasoning steps?

The system is split into two main stages:

1. **Stage 1: Candidate generation**
   - Each active agent solves the task multiple times.
   - Each run must produce explicit reasoning steps and a short final answer.
   - If tool use is enabled, an agent may request tools through a structured JSON format.
   - Answers are normalized and grouped to compute confidence.

2. **Stage 2: Reasoning Verifying**
   - Candidate reasoning steps are judged by other agents.
   - Judge agents return scores only.
   - Tool evidence can be included so judges can penalize unsupported or conflicting reasoning.
   - Final ranking combines confidence, judge score, and rule-based penalty.

Tools are only used before or during Stage 1. After Stage 1 self-consistency
aggregation is complete, SCP does not call search, calculator, solver, or
attachment tools again. Stage 2 only judges the recorded answers, reasoning
steps, and tool evidence.

## Architecture

```text
Question
   |
   v
Evidence Preparation
   |-- attachment handling
   |-- optional search preparation
   |-- optional solver/calculator preparation
   |-- ContextPacket construction
   |
   v
Stage 1 Runner
   |-- Agent A x N runs, optional per-run tool trajectory
   |-- Agent B x N runs, optional per-run tool trajectory
   |-- Agent C x N runs, optional per-run tool trajectory
   |-- Agent D x N runs, optional per-run tool trajectory
   |
   v
Stage 1 Aggregation
   |-- normalized answer grouping
   |-- confidence score
   |-- early-stop decision, if enabled
   |
   v
Stage 2 Runner
   |-- pairwise judge calls
   |-- reasoning-step score parsing
   |-- tool evidence aware scoring
   |
   v
Final Selection
   |-- confidence_score
   |-- avg_judge_score
   |-- penalty_score
   |-- total_score
   |
   v
Answer + JSON/Markdown Reports
```

## Repository Structure

```text
AgentConsis-Verify/
├── benchmark/
│   └── gaia/                 # GAIA dataset loader, runner, report/export helpers
├── context/                  # Prompt and context builders
├── core/                     # Main orchestration, runners, config, SLM agent wrapper
├── data/                     # Local dataset/cache-related files
├── outputs/                  # Generated experiment outputs
├── parsers/                  # Reasoning, tool request, stage output, and judge parsers
├── score/                    # Aggregation, validation, scoring, and penalties
├── tools/                    # Tool manager, cache, search, calculator, attachment tools
├── utils/                    # Shared utilities
├── exceptions.py
├── run_gaia.py               # Main GAIA CLI entry point
└── README.md
```

## Installation

Create and activate a Python environment:

```bash
python -m venv venv312
venv312\Scripts\activate
```

Install the project dependencies according to your local environment.
This project expects local model serving through an OpenAI-compatible endpoint,
such as Ollama-compatible APIs.

Dependencies for embedding-delta search query planning:

```bash
pip install transformers torch
```

The query planner uses a HuggingFace encoder / embedding model for
representation sensitivity analysis. Each candidate token is removed from the
question, the query embedding is recomputed, and the embedding delta is used as
the salience signal. The current default is:

```text
BAAI/bge-m3
```

This model is loaded when search evidence needs query planning.

## Environment Variables

Create a `.env` file in the project root. Do not commit real secrets.

Common model/API settings:

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=
OLLAMA_TIMEOUT=120

Nemotron_MODEL_ID=nemotron-mini:4b
Minicpm_MODEL_ID=yefx/minicpm3_4b
Qwen_MODEL_ID=qwen3:4b
Gemma_MODEL_ID=gemma3:4b
```

The default model aliases used by the GAIA runner are mapped by `SLM_Agent` to
the environment variables above:

| Internal alias | Environment variable | Example local model |
| --- | --- | --- |
| `nemotron-mini:4b` | `Nemotron_MODEL_ID` | `nemotron-3-nano:4b` |
| `minicpm3:4b` | `Minicpm_MODEL_ID` | `yefx/minicpm3_4b` |
| `qwen3:4b` | `Qwen_MODEL_ID` | `qwen3:4b` |
| `gemma3:4b` | `Gemma_MODEL_ID` | `gemma4:e4b` |

Optional vision and attachment settings:

```env
OLLAMA_VISION_MODEL=your_vision_model
OLLAMA_VISION_TIMEOUT=180
```

Optional search settings:

```env
SEARCH_BACKEND=searxng
SEARXNG_URL=http://localhost:8080
SEARCH_SALIENCE_HF_MODEL=BAAI/bge-m3
TAVILY_API_KEY=
SERPAPI_API_KEY=
PERPLEXITY_API_KEY=
```

Supported search backends include `hybrid`, `advanced`, `searxng`, `tavily`,
`serpapi`, `duckduckgo`, and `perplexity`. `SearchTool` only returns normalized
raw search results. Source filtering, full-page fetch, chunk labeling,
deduplication, retrieval control, next-hop query generation, and prompt
rendering are controlled by `tools/search_result_builder/`.

Helpfulness probability scoring is currently disabled in the active search
pipeline. The project keeps the standalone helper implementation, but evidence
conversion does not use helpfulness probability while the search flow is being
simplified and the EfficientRAG labeler model is still pending.

GAIA dataset access may require:

```env
HF_TOKEN=
```

## Quick Start

Run one GAIA Level 1 example:

```bash
python run_gaia.py --level 1 --max-samples 1 --log-name gaia_level1_test
```

Run one GAIA Level 2 example with Stage 1 tool use:

```bash
python run_gaia.py ^
  --level 2 ^
  --max-samples 1 ^
  --enable-stage1-tool-use ^
  --max-stage1-tool-turns 2 ^
  --log-name gaia_level2_tool_test
```

Run one GAIA example with the default evidence-driven search flow:

```bash
python run_gaia.py ^
  --level 2 ^
  --max-samples 1 ^
  --log-name gaia_level2_evidence_search_test
```

Run the full GAIA Level 1 validation split:

```bash
python run_gaia.py --level 1 --log-name gaia_level1_complete
```

## CLI Usage

```bash
python run_gaia.py \
  --data-dir data \
  --split validation \
  --level 1 \
  --max-samples 1 \
  --stage1-runs-per-agent 3 \
  --stage2-max-tokens 64 \
  --temperature 0.2 \
  --log-name experiment_name \
  --output-dir outputs
```

Important options:

| Argument | Description |
| --- | --- |
| `--data-dir` | Local dataset/cache directory. |
| `--split` | GAIA split, usually `validation` or `test`. |
| `--level` | GAIA level: `1`, `2`, or `3`. |
| `--max-samples` | Limit number of tasks for smoke tests. |
| `--stage1-runs-per-agent` | Number of Stage 1 attempts per agent. |
| `--stage2-max-tokens` | Maximum tokens for Stage 2 judge responses. |
| `--enable-stage1-tool-use` | Enable tool trajectory during Stage 1. |
| `--max-stage1-tool-turns` | Maximum tool-use turns per Stage 1 run. |
| `--enable-evidence-driven-search` / `--no-enable-evidence-driven-search` | Enable or disable search_result_builder next-hop retrieval. Enabled by default. |
| `--enable-stage1-early-stop` | Enable early stopping during Stage 1. |
| `--stage1-early-stop-max-retries` | Retry budget for early-stop mode. |
| `--models` | Comma-separated model list. |
| `--log-name` | Name of the experiment output folder/report. |
| `--output-dir` | Root directory for exported results. |
| `--report-md` | Optional custom Markdown report path. |

## Default Agents

The default GAIA runner configures four agents:

| Agent | Default model |
| --- | --- |
| `nemotron` | `nemotron-mini:4b` |
| `minicpm` | `minicpm3:4b` |
| `qwen` | `qwen3:4b` |
| `gemma` | `gemma3:4b` |

Each agent is wrapped by `SLM_Agent`, which calls the configured local
OpenAI-compatible model endpoint.

## Stage 1 Output Format

Agents are expected to return explicit reasoning steps and a short final answer:

```text
REASONING =
step 1. ...
step 2. ...
step 3. ...
FINAL_ANSWER = ...
```

When Stage 1 tool use is enabled, agents may request a tool through JSON:

```json
{
  "type": "tool_request",
  "reasoning_step": "step 1. I need to verify the relevant date.",
  "tool_name": "search",
  "tool_args": {
    "input": "query text"
  }
}
```

After enough evidence is collected, the agent must return:

```json
{
  "type": "final_answer",
  "reasoning": "step 1. ...\nstep 2. ...",
  "final_answer": "short final answer only"
}
```

## Tool Use

Currently supported Stage 1 tools include:

| Tool | Purpose |
| --- | --- |
| `search` | Call a configured web search backend and return normalized raw results. |
| `python_calculator` | Compute deterministic arithmetic or small calculations. |

Tool calls are cached by normalized tool name and arguments. If two runs ask for
the same tool with the same arguments, SCP reuses the cached result.

There are two tool-use windows:

1. **Shared evidence preparation before Stage 1**
   - `EvidenceRunner` may route the task to search, attachment handling,
     deterministic solver, or calculator before any agent starts reasoning.
   - The resulting evidence is shared with all Stage 1 agents.
   - Routing first checks closed-world, deterministic, attachment, and puzzle
     signals before factual search. Weak question words such as `who`, `which`,
     `when`, and `where` do not trigger search by themselves.

2. **Optional per-run Stage 1 tool trajectory**
   - Enabled by `--enable-stage1-tool-use`.
   - Each agent run may request tools up to `--max-stage1-tool-turns`.
   - Tool requests must use the structured JSON format described above.

After all Stage 1 runs are aggregated, SCP does not use tools again. Stage 2
judges only consume the target answer, target reasoning, and recorded tool
evidence.

The shared search flow uses `EvidenceSearcher`:

```text
question
-> embedding-delta query planning
-> SearchTool backend call
-> source hard filter
-> full-page fetch
-> chunking
-> EfficientRAG labeler fallback
-> SEER n-gram dedup
-> evidence conversion
-> retrieval control
-> optional EfficientRAG-filtered next-hop query
-> structured evidence rendering
```

`SearchTool` is intentionally small. It does not rerank, fetch full pages,
filter sources, or build evidence. It only returns backend results with this
shape:

```python
{
    "results": [
        {"title": "...", "url": "...", "content": "..."}
    ],
    "backend": "searxng",
    "answer": None,
    "notices": []
}
```

`search_result_builder` owns the rest of the search data flow.

Query planning uses `QueryGenerator`, backed by
`MaskSalienceQueryGenerator`. The generator performs representation
sensitivity analysis:

```text
question
-> HF tokenizer + encoder model
-> delete one token/span
-> recompute sentence embedding
-> score by embedding delta
-> select the top 5 semantic-impact spans for search-enabled tasks
-> ask qwen3:4b for concise query candidates
-> clean and deduplicate generated queries while preserving order
```

Closed-world, deterministic, attachment, and puzzle-like tasks are filtered by
routing before search query planning, so top-5 span expansion is only used when
the system has already decided that factual search evidence is appropriate.

The current search pipeline components are:

| Component | Role |
| --- | --- |
| `query/` | Builds first-hop search queries from embedding-delta salient spans. |
| `source_analyze/seer/source_filter.py` | Removes leaks, duplicates, low-value pages, and question echoes. |
| `source_analyze/seer/page_content_fetcher.py` | Fetches full pages after source filtering. |
| `source_analyze/rag_labeler.py` | Labels chunks as useful/useless. Currently uses deterministic fallback until the trained labeler is connected. |
| `source_analyze/seer/helpfulness_expert.py` | Standalone probability helper. Not used by the active pipeline right now. |
| `source_analyze/seer/ngram_deduplicate.py` | Deduplicates useful evidence chunks. |
| `next_hop_query/retrieval_controller.py` | Decides whether next-hop retrieval is needed. |
| `next_hop_query/rag_filter.py` | Builds an EfficientRAG-style filtered follow-up query when retrieval control requests next-hop. |
| `next_hop_query/next_hop_query_generator.py` | Builds follow-up queries from evidence and question signals. |
| `evidence_renderer.py` | Renders compact structured context for agents. |

The prompt sent to agents is intentionally compact:

```text
Original Question:
...

Query:
[Q1] ...

Evidence:
[E1]
Source: ...
Query: Q1
Text: ...
```

Candidate-answer extraction is currently not required for the agent prompt:
agents receive structured evidence directly. Search diagnostics are exported in
per-task JSON so each run can show source filtering, fetch, chunking, labeling,
deduplication, retrieval control, and next-hop decisions.

## Scoring

### Confidence Score

Stage 1 confidence is based on normalized answer agreement across repeated runs:

| Agreement pattern | `confidence_score` |
| --- | ---: |
| 3 answers are the same | `1.00` |
| 2 answers are the same | `0.67` |
| 3 answers are different | `0.33` |




## Outputs

Each experiment writes to:

```text
outputs/{log_name}/
├── tasks/
│   ├── task_0001.json
│   ├── task_0002.json
│   └── ...
└── {log_name}.md
```

Per-task JSON files contain task metadata, predictions, scores, timing,
token usage when available, tool summaries, search pipeline diagnostics, and
intermediate stage results.

The Markdown report summarizes:

- run configuration
- per-task result table
- final answers
- accuracy statistics
- response time
- token usage
- tool-use summary



## Development Notes

Recommended smoke test:

```bash
python run_gaia.py --level 1 --max-samples 1 --log-name smoke_test
```

Recommended tool-use smoke test:

```bash
python run_gaia.py --level 2 --max-samples 1 --enable-stage1-tool-use --log-name smoke_tool_test
```

Recommended evidence-search smoke test:

```bash
python run_gaia.py --level 2 --max-samples 1 --log-name smoke_evidence_search
```

Useful checks:

```bash
python run_gaia.py --help
```



