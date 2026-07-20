<div align="center">

# SCP

### Small-model Collaborative Pipeline

**Multi-agent, evidence-grounded reasoning for GAIA-style tasks — built entirely on small local language models.**

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20cuda-lightgrey)](#installation)
[![Status](https://img.shields.io/badge/status-experimental%20research-orange)](#project-status)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#development)
[![License](https://img.shields.io/badge/license-unspecified-lightgrey)](#license)

[Overview](#overview) · [Pipeline](#pipeline) · [Quick Start](#quick-start) · [CLI Options](#cli-options) · [Architecture](#architecture-deep-dive) · [Outputs](#outputs) · [Limitations](#current-limitations) · [Contributing](#contributing)

</div>

---

## Overview

SCP asks a simple question: **how far can a team of small local language models get on hard, multi-step reasoning tasks if you invest the effort in evidence quality, verification, and answer selection instead of model size?**

Instead of one large model answering once, SCP runs several small models (4B-class, served locally through Ollama) multiple times each, grounds every run in retrieved or parsed evidence, scores every reasoning step with a dedicated process-reward model, and only then decides on a final answer — through an ordered set of gates rather than a single weighted score.

SCP targets [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)-style benchmark tasks: questions that require multi-hop web research, attachment/spreadsheet/video comprehension, deterministic computation, or some combination of all three.

**Design principles:**

- ✅ **Evidence before reasoning** — search, attachments, video, tables, and deterministic handlers are converted into traceable evidence *before* any agent has to answer.
- ✅ **Small models, many votes** — each agent answers several times; self-consistency is aggregated per agent before anything is compared across agents.
- ✅ **Tool-aware reasoning** — agents can request structured tool calls mid-reasoning, under an explicit turn budget and anti-repetition policy.
- ✅ **A Fact Store, not just text** — search, attachment, handler, and tool results are normalized into a shared fact store so later stages can verify claims, not just re-read text.
- ✅ **Process verification, not outcome guessing** — Stage 2 scores *each reasoning step* with a process-reward model instead of asking a judge model to grade the final answer.
- ✅ **Ordered gates, not one score** — the final answer is decided by passing candidates through validity → requirement → contradiction → evidence support → consensus → self-consistency → verification, in that order.

---

## Pipeline

```text
GAIA Task
  |
  v
Evidence Prepare
  |-- metadata / system routing
  |-- attachment parsing
  |-- deterministic handlers
  |-- web search and retrieval
  |-- Fact Store construction
  |
  v
Stage 1: Multi-Agent Reasoning
  |-- nemotron x N runs
  |-- qwen x N runs
  |-- gemma x N runs
  |-- optional tool-use trajectory
  |-- per-agent self-consistency
  |
  v
Stage 2: VersaPRM Process Verification
  |-- parse reasoning into steps
  |-- score each reasoning step
  |-- compute critical-step probability
  |
  v
Final Winner Selection (ordered gates)
  |-- answer validity
  |-- answer requirement check
  |-- contradiction check
  |-- evidence support check
  |-- cross-agent consensus
  |-- self-consistency
  |-- VersaPRM verification
  |
  v
Answer + JSON / Markdown reports
```

---

## Quick Start

```bash
git clone https://github.com/j2682j/Self_Consistency_with_Scoring_for_SLM_in_Multi_agent_system.git
cd Self_Consistency_with_Scoring_for_SLM_in_Multi_agent_system

python -m venv venv312
venv312\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Run a smoke test on a single GAIA task:

```bash
python run_gaia.py --split validation --level 1 --max-samples 1 --log-name smoke_level1
```

Run a full experiment:

```bash
python run_gaia.py ^
  --split validation ^
  --level 1 ^
  --max-samples 60 ^
  --stage1-runs-per-agent 3 ^
  --max-stage1-workers 1 ^
  --temperature 0.3 ^
  --evidence-prepare ^
  --enable-evidence-driven-search ^
  --enable-stage1-tool-use ^
  --max-stage1-tool-turns 5 ^
  --stage1-prepared-search-budget 1 ^
  --max-parallel-next-hop-queries 2 ^
  --versa-prm-local-files-only ^
  --log-name level1_full_system_final
```

Using the project virtual environment directly from Git Bash / MSYS:

```bash
/c/SCP/venv312/Scripts/python.exe run_gaia.py --split validation --level 1 --max-samples 1 --log-name smoke_level1
```

> SCP is developed and tested on Windows with an NVIDIA CUDA GPU. See [Installation](#installation) for platform notes.

---

## Default Agents

| Agent id | Default model |
| --- | --- |
| `nemotron` | `nemotron-3-nano:4b` |
| `qwen` | `qwen3:4b` |
| `gemma` | `gemma3:4b` |

Defaults live in `benchmark/gaia/gaia_runner.py` as `DEFAULT_AGENT_SPECS` and can be overridden from the CLI:

```bash
python run_gaia.py --level 1 --max-samples 1 --models qwen3:4b --log-name qwen_only_level1
```

---

## Installation

Python 3.12 is recommended.

```bash
python -m venv venv312
venv312\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

External, non-Python services used by the full pipeline:

| Service | Purpose |
| --- | --- |
| [Ollama](https://ollama.com/) | Serves the local Stage 1 SLMs and the vision model. |
| [SearXNG](https://docs.searxng.org/) | Local metasearch backend for web retrieval. |
| FFmpeg / ffprobe | Media processing for audio/video attachments. |
| [Stockfish](https://stockfishchess.org/) | Chess-related deterministic handlers. |
| Hugging Face cache | GAIA dataset and VersaPRM model weights. |

**Core dependencies:** `torch` + `transformers` + `peft` (local inference and VersaPRM), `sentence-transformers` + `faiss-cpu` (retrieval), `spaCy` (NLP), `trafilatura` / `playwright` / `beautifulsoup4` (web extraction), `pandas` / `openpyxl` / `python-docx` / `python-pptx` / `PyMuPDF` (attachment parsing), `python-chess`, `yt-dlp` / `youtube-transcript-api` / `faster-whisper` (video/audio). See [`requirements.txt`](requirements.txt) for the full pinned list.

### Environment

Create a `.env` file in the project root — keep real tokens out of git.

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_TIMEOUT=120

Nemotron_MODEL_ID=nemotron-3-nano:4b
Qwen_MODEL_ID=qwen3:4b
Gemma_MODEL_ID=gemma3:4b
QUERY_GENERATOR_MODEL=qwen3:4b

OLLAMA_VISION_MODEL=qwen3-vl:4b
OLLAMA_VISION_TIMEOUT=180

SEARCH_BACKEND=searxng
SEARXNG_URL=http://localhost:8080
SEARCH_SALIENCE_HF_MODEL=BAAI/bge-m3

VERSA_PRM_MODEL=UW-Madison-Lee-Lab/VersaPRM-Base-3B
VERSA_PRM_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
VERSA_PRM_DEVICE=auto
VERSA_PRM_DTYPE=auto
VERSA_PRM_LOCAL_FILES_ONLY=true

HF_TOKEN=
```

- VersaPRM defaults to local Hugging Face cache usage — pass `--versa-prm-allow-download` only when you intentionally want network downloads.
- Some gated Hugging Face models require accepted access and an authenticated token.

---

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `data/gaia` | GAIA dataset cache directory. |
| `--split` | `validation` | Dataset split: `validation` or `test`. |
| `--level` | `None` | GAIA level: `1`, `2`, or `3`. |
| `--max-samples` | `1` | Number of samples to run. |
| `--stage1-runs-per-agent` | `3` | Repeated runs per Stage1 agent. |
| `--max-stage1-workers` | automatic | Max parallel Stage1 workers. Use `1` to reduce VRAM pressure. |
| `--temperature` | `0.5` | Stage1 model sampling temperature. |
| `--models` | default agents | Comma-separated model list. |
| `--evidence-prepare` / `--no-evidence-prepare` | enabled | Enable or disable pre-Stage1 evidence preparation. |
| `--enable-evidence-driven-search` | enabled | Enable the web retrieval pipeline when search is needed. |
| `--bypass-search-labeler` | disabled | Skip the EfficientRAG labeler and build sentence-level evidence units directly. |
| `--enable-stage1-tool-use` | disabled | Allow Stage1 agents to request tools during reasoning. |
| `--max-stage1-tool-turns` | `2` | Base Stage1 tool-use turn budget. |
| `--stage1-prepared-search-budget` | `1` | Supplemental Stage1 search budget when prepared search evidence already exists. |
| `--without-stage2-score` | disabled | Skip VersaPRM scoring and select based on Stage1 metadata only. |
| `--stage2-verifier` | `versa` | Stage2 verifier backend. Currently only `versa` is supported. |
| `--versa-prm-local-files-only` | enabled | Load VersaPRM from local Hugging Face cache only. |
| `--versa-prm-allow-download` | disabled | Allow Hugging Face downloads for VersaPRM. |
| `--log-name` | `gaia_run` | Output folder and Markdown report name. |
| `--output-dir` | auto | Override per-task JSON output directory. |
| `--report-md` | auto | Override Markdown report path. |

---

## Architecture Deep Dive

### Evidence Prepare

Runs before Stage 1 when `--evidence-prepare` is enabled:

1. **System routing** — inspects metadata and question signals to separate attachment, deterministic, factual-search, video/media, and fallback tasks, producing a route contract and an evidence-readiness state.
2. **Attachment parsing** — builds an attachment profile and reads text, Office files, spreadsheets, archives, PDFs, images, audio, and video, converting the result into compact evidence and semantic facts.
3. **Deterministic handlers** — registered handlers cover exact tasks such as tables, graph search, coordinate distance, string transforms, chess, grid/path puzzles, unit conversion, date/time, and numeric reasoning, each behind an input/output contract and a trust gate.
4. **Web search and retrieval** — generates queries from semantic-impact spans and question-role information, calls a search backend (usually SearXNG), filters unsafe or low-quality sources, fetches full pages when needed, builds a temporary per-task corpus, retrieves passages via multilingual-E5 embeddings + FAISS, and converts useful material into evidence items and facts.
5. **Fact Store construction** — stores direct, bridge, derived, and negative/absence facts with provenance. Evidence readiness is decided from *verifiable facts*, not from raw text availability alone.

### Search Flow

Centered on `tools/search_result_builder/retrieval_control.py`:

```text
Question
  -> semantic-impact span extraction
  -> query generation
  -> search backend (SearXNG by default)
  -> source safety filter
  -> full-page fetch or browser fetch when needed
  -> page cleaning and chunking
  -> temporary corpus
  -> multilingual-e5-base passage embeddings
  -> FAISS top-k retrieval
  -> labeler or bypass evidence-unit construction
  -> span recovery and role classification
  -> direct / bridge evidence contract
  -> next-hop query composition when goals remain incomplete
  -> EvidenceItems + Fact Store
```

The search backend adapter itself stays intentionally small — `tools/search_tool.py` only calls the configured backend and returns normalized raw results:

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

All filtering, fetching, corpus building, FAISS retrieval, evidence conversion, next-hop recovery, and diagnostics live under `tools/search_result_builder/`.

### Attachment Flow

Attachment evidence is handled in two windows:

| Window | Behavior |
| --- | --- |
| **Evidence Prepare** | `AttachmentReaderTool` parses the file before Stage 1; `AttachmentEvidenceBuilder` formats the content; semantic fact extraction stores useful relations; the result is shared by every Stage 1 agent. |
| **Stage 1 Tool Use** | An agent can request `attachment_reader` mid-reasoning if enabled; a shared workspace prevents re-parsing the same file; outputs are recorded as Stage 1 tool evidence and may support final selection later. |

Supported reader families:

| Type | Handling |
| --- | --- |
| Text / code | Plain text extraction and preview. |
| PDF | Text extraction via PDF readers / PyMuPDF. |
| Office | Word, PowerPoint, spreadsheet parsing. |
| Image | Vision model and specialized visual extractors where applicable. |
| Audio / video | Transcript and media evidence tools. |
| Archive | Safe unpacking and recursive profile construction. |
| Specialized | Chess board and fraction-document extractors. |

### Stage 1: Multi-Agent Reasoning

Each active SLM agent answers the task several times. Every run receives the original question, the answer requirement, compact evidence context, attachment metadata (if any), tool availability and policy (if enabled), and the prior tool trace within the same trajectory.

Expected final-answer shape:

```json
{
  "type": "final_answer",
  "reasoning_steps": ["step 1. ...", "step 2. ..."],
  "final_answer": "short answer only",
  "confidence": 0.0,
  "used_evidence_ids": [],
  "answer_type": "short_text",
  "tool_request": null
}
```

Tool request shape:

```json
{
  "type": "tool_request",
  "reasoning_step": "step 1. I need to inspect the spreadsheet.",
  "tool_name": "attachment_reader",
  "tool_args": {
    "question": "...",
    "file_path": "..."
  }
}
```

Per-agent self-consistency confidence:

| Agreement within one agent | Confidence |
| --- | ---: |
| 3 identical answers | `1.00` |
| 2 identical answers | `0.67` |
| all different | `0.33` |

### Stage 2: VersaPRM Process Verification

Stage 2 treats correctness as a property of the *reasoning process*, not just the final token:

1. Parse each candidate's reasoning into canonical steps.
2. Strip final-answer leakage out of the reasoning steps.
3. Send only the question and the reasoning steps to VersaPRM.
4. Record one reward probability per step.
5. Compute the average reward probability, the critical-step floor (the weakest step), and the critical-step geometric mean.

VersaPRM is lazy-loaded and unloaded between scoring phases to reduce VRAM pressure.

```text
Verifier model:      UW-Madison-Lee-Lab/VersaPRM-Base-3B
PEFT fallback base:  meta-llama/Llama-3.2-3B-Instruct
```

```bash
python score/test_versa_load.py --local-files-only
```

### Final Winner Selection

SCP does **not** pick a winner by summing confidence, judge score, and penalty into one number. Candidates instead pass through ordered gates, each of which can eliminate, defer, or narrow the field — evidence and answer-requirement checks dominate consensus and consistency, so a highly self-consistent answer can still lose if it conflicts with evidence or fails the task's answer requirement:

```text
Candidate answers
  -> Validity Gate
  -> Answer Requirement Gate
  -> Contradiction Gate
  -> Evidence Support Gate
  -> Cross-Agent Consensus Gate
  -> Self-Consistency Gate
  -> VersaPRM Verification Gate
  -> Final answer
```

Evidence support is ranked ordinally, not weighted:

```text
contradicted < unsupported < bridge_evidence < direct_evidence < verified_derived < trusted_tool_final
```

---

## Outputs

Each run writes:

```text
outputs/{log_name}/
  tasks/
    001_{task_id}.json
    002_{task_id}.json
    ...
  {log_name}.md
```

Per-task JSON includes the question, expected/predicted answer, exact/partial match, response time and token usage, every Stage 1 run and its reasoning steps, VersaPRM step probabilities, the final winner gate trace, evidence support results, routing and evidence-readiness state, tool-usage trace, search diagnostics, and a Fact Store snapshot.

The Markdown report summarizes run configuration, the per-task result table, accuracy, average token usage, average response time, and search/tool/evidence summaries.

---

## Repository Layout

```text
SCP/
  benchmark/
    gaia/                         GAIA dataset loading, evaluation, reports
  context/                        Stage1 / Stage2 context builders
  core/
    network.py                    Main orchestration
    evidence_runner.py            Evidence Prepare orchestration
    stage1_runner.py              Multi-agent Stage1 runner
    stage1_tool_use_runner.py     Optional Stage1 tool trajectory
    stage2_runner.py              VersaPRM step verifier
    candidate_path_evaluator.py   Candidate-level support + Versa evaluation
    config.py                     Shared dataclasses
  parsers/
    reasoning_parser.py           Reasoning step parser for VersaPRM
    tool_request_parser.py        Structured tool request parser
  score/
    answer_validator.py           Final answer validation and repair helpers
    answer_candidate_clusterer.py Candidate clustering
    evidence_support_checker.py   Candidate-to-evidence verification
    final_winner_selector.py      Ordered gate winner selection
    versa_prm_scorer.py           VersaPRM model wrapper
  tools/
    tool_manager.py               Tool registry and execution trace
    search_tool.py                Search backend adapter
    attachment_reader/            Attachment parsing pipeline
    deterministic_handlers/       Deterministic solver contracts and handlers
    evidence/                     Fact Store and semantic fact extraction
    search_result_builder/        Web retrieval, corpus, FAISS, evidence conversion
  outputs/                        Experiment outputs
  tests/                          Unit and integration tests
  run_gaia.py                     Main CLI entry
```

---

## Local Services

**Ollama** — confirm the required models are pulled:

```bash
ollama list
```

```text
nemotron-3-nano:4b
qwen3:4b
gemma3:4b
qwen3-vl:4b
```

**SearXNG** — the search backend expects an instance at `http://localhost:8080`. SearXNG is used purely as a metasearch backend; SCP performs its own source filtering, page fetching, corpus construction, embedding retrieval, and evidence conversion after receiving results.

---

## Development

Run the full test suite:

```bash
python -m pytest
```

Run focused tests:

```bash
python -m pytest tests/test_evidence_readiness.py tests/test_content_requirement.py -q
```

Compile check:

```bash
python -m compileall core score tools parsers benchmark
```

---

## Current Limitations

- Level 3 tasks often require source-type-specific retrieval, full-document extraction, database/API access, video understanding, or structured collection reasoning that the current pipeline does not fully cover.
- Search can retrieve relevant pages but still fail to convert them into direct answer-support facts.
- EfficientRAG Labeler quality strongly affects span and next-hop quality; `--bypass-search-labeler` is available for diagnostic experiments.
- Vision and video evidence can be expensive in VRAM — keep `--max-stage1-workers 1` on constrained local GPUs.
- VersaPRM requires local cache access to gated Hugging Face models unless download mode is explicitly enabled.

---

## Project Status

SCP is an experimental research system. The codebase is optimized for iterative GAIA experiments, trace inspection, and ablation studies — not for production deployment.

## Contributing

This is an active research codebase. Issues, ablation results, and pull requests that come with a reproducible `run_gaia.py` command and an output log are especially welcome.

## License

No license file is currently published in this repository. Until one is added, please treat the code as **all rights reserved** and contact the maintainer before reuse or redistribution.

## Contact

Maintained by [@j2682j](https://github.com/j2682j). Open an issue for questions, bug reports, or experiment discussion.
