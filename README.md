<div align="center">

# SCP

**Self-Consistent Peer-scored multi-agent reasoning for GAIA**

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Benchmark](https://img.shields.io/badge/Benchmark-GAIA-purple)
![Agents](https://img.shields.io/badge/Multi--Agent-SLM-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

</div>

## TL;DR

SCP is a local multi-agent reasoning framework for GAIA-style question answering.
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

## Key Features

- **Multi-agent Stage 1 reasoning**: multiple SLM agents answer the same task in parallel.
- **Self-consistency scoring**: repeated answers are normalized and grouped into confidence scores.
- **Optional tool use**: Stage 1 agents can request tools before producing a final answer.
- **Tool cache**: identical tool calls reuse cached results.
- **Cross-agent Stage 2 judging**: agents judge other agents' reasoning steps.
- **Step-level scoring**: judge score is computed over explicit reasoning steps.
- **Rule-based penalties**: malformed answers, tool-call-as-answer, refusal-like answers, and tool failures can reduce score.
- **GAIA-oriented evaluation**: includes dataset loading, attachment preparation, batch running, per-task export, Markdown reporting, and accuracy statistics.

## Overview

SCP is designed around the following question:

> Can a group of small local agents improve task reliability by reasoning independently, comparing answers, and judging each other's reasoning steps?

The system is split into two main stages:

1. **Stage 1: Candidate generation**
   - Each active agent solves the task multiple times.
   - Each run must produce explicit reasoning steps and a short final answer.
   - If tool use is enabled, an agent may request tools through a structured JSON format.
   - Answers are normalized and grouped to compute confidence.

2. **Stage 2: Peer judging**
   - Candidate reasoning steps are judged by other agents.
   - Judge agents return scores only.
   - Tool evidence can be included so judges can penalize unsupported or conflicting reasoning.
   - Final ranking combines confidence, judge score, and rule-based penalty.

## Architecture

```text
Question
   |
   v
Evidence Preparation
   |-- attachment handling
   |-- optional search preparation
   |-- ContextPacket construction
   |
   v
Stage 1 Runner
   |-- Agent A x N runs
   |-- Agent B x N runs
   |-- Agent C x N runs
   |-- Agent D x N runs
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
SCP/
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
TAVILY_API_KEY=
SERPAPI_API_KEY=
PERPLEXITY_API_KEY=
```

Supported search backends include `hybrid`, `advanced`, `searxng`, `tavily`,
`serpapi`, `duckduckgo`, and `perplexity`. The search tool can return structured
results and, when precision is needed, conditionally fetch full page content for
evidence extraction.

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
    "input": "query text",
    "mode": "text"
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
| `search` | Build evidence-oriented web search context. |
| `python_calculator` | Compute deterministic arithmetic or small calculations. |

Tool calls are cached by normalized tool name and arguments. If two runs ask for
the same tool with the same arguments, SCP reuses the cached result.

The shared search flow uses `EvidenceSearcher`:

```text
question
-> query planning
-> structured search
-> source filtering
-> evidence extraction
-> candidate answer extraction
-> prompt rendering
```

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

Candidate Answer:
[C1] answer=...; type=...; evidence=E1
```

URLs, filtered sources, relevance scores, and other diagnostics are kept in the
exported `raw_result` for debugging, but are not included in the agent prompt.

## Scoring

### Confidence Score

Stage 1 confidence is based on normalized answer agreement across repeated runs:

| Agreement pattern | `confidence_score` |
| --- | ---: |
| 3 answers are the same | `1.00` |
| 2 answers are the same | `0.67` |
| 3 answers are different | `0.33` |

### Stage 2 Judge Score

Stage 2 judges each reasoning step with a score from `-1.0` to `1.0`:

| Score | Meaning |
| --- | --- |
| `1.0` | Correct, useful, and supported by question/evidence/tool result. |
| `0.5` | Mostly correct but incomplete or partially supported. |
| `0.0` | Unclear, redundant, or impossible to judge. |
| `-0.5` | Unsupported, skips an important check, or contains a weak conflict. |
| `-1.0` | Contradicts evidence, invents evidence, misuses tool result, or supports a wrong/malformed answer. |

Judge agents must return scores only:

```json
{
  "step_scores": [
    {"step": 1, "score": 1.0},
    {"step": 2, "score": 0.5}
  ]
}
```

### Rule-based Penalty

The penalty calculator can reduce a candidate score for issues such as:

- malformed or missing final answer
- tool-call JSON used as final answer
- refusal-like answer such as `unknown` or `insufficient data`
- failed tool calls
- all tool calls failing
- tool failure immediately before final answer

The final ranking score is:

```text
total_score = confidence_score + avg_judge_score + penalty_score
```

## Early Stop

When Stage 1 early-stop mode is enabled:

- If the best candidate reaches `confidence_score = 1.0`, SCP can return it directly.
- If the best candidate reaches `confidence_score = 0.67`, SCP asks the previous best agent to judge the candidate reasoning steps.
- If the average judge score is greater than `0`, SCP accepts the candidate.
- Otherwise SCP retries Stage 1 within the configured retry budget.

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
token usage when available, tool summaries, and intermediate stage results.

The Markdown report summarizes:

- run configuration
- per-task result table
- final answers
- accuracy statistics
- response time
- token usage
- tool-use summary

## Main Modules

| Module | Responsibility |
| --- | --- |
| `core/network.py` | Top-level orchestration of evidence, Stage 1, Stage 2, and final selection. |
| `core/evidence_runner.py` | Prepares attachments, search evidence, and context packets. |
| `core/stage1_runner.py` | Runs parallel Stage 1 agent attempts. |
| `core/stage1_trajectory_runner.py` | Runs iterative Stage 1 tool-use trajectories. |
| `core/stage2_runner.py` | Runs cross-agent judge scoring. |
| `core/slm_agent.py` | Calls local OpenAI-compatible SLM endpoints. |
| `context/` | Builds prompts and agent context. |
| `parsers/` | Parses reasoning steps, tool requests, final answers, and judge outputs. |
| `score/` | Computes confidence, validation, penalties, and final scores. |
| `tools/` | Provides search, calculation, attachment handling, and tool caching. |

## Development Notes

Recommended smoke test:

```bash
python run_gaia.py --level 1 --max-samples 1 --log-name smoke_test
```

Recommended tool-use smoke test:

```bash
python run_gaia.py --level 2 --max-samples 1 --enable-stage1-tool-use --log-name smoke_tool_test
```

Useful checks:

```bash
python run_gaia.py --help
```

## Limitations

- The project is experimental and optimized for iteration, not production serving.
- Model quality depends strongly on the local SLMs and endpoint configuration.
- Some GAIA tasks may require richer tool routing, better attachment understanding, or stronger semantic answer equivalence.
- Web search quality depends on the configured backend.
- Token usage is reported only when the model endpoint returns usage metadata.

## Acknowledgements

This README structure is inspired by the style of
[LINs-lab/SupervisorAgent](https://github.com/LINs-lab/SupervisorAgent).

SCP is developed as a local experimental framework for studying small-model
multi-agent reasoning, self-consistency, tool use, and peer judging on GAIA.
