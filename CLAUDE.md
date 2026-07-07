# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Official implementation of *"The Pragmatic Persona: Discovering LLM Persona through Bridging Inference"* (ICPR 2026). The **PD-Agent** interviews a target LLM, extracts implicit "bridging" discourse relations from the dialogue, builds a semantic graph, and infers the target's hidden persona across four dimensions (Social Role, Big-Five Personality, Background, Interests). Predicted persona is scored against the injected ground-truth via embedding cosine similarity.

## Current Direction

Active work extends this ICPR paper into an **IEEE Access** journal version. Ongoing tasks, priorities, and the target contribution live in **`ROADMAP.md`** — read it before starting feature work so changes align with the journal extension (rigorous ablation + honest per-dimension evaluation first; BIG-Net graph representation and causal validation next).

## Environment & Setup

- Python 3.10+.
- `pip install -r requirements.txt` (torch, transformers, langchain/langgraph, networkx, streamlit).
- Secrets live in a **`.env` at the repo root** (git-ignored, manually parsed by each script's `load_env_file()` — not python-dotenv):
  - `OPENAI_API_KEY` — required for the GPT tool/reasoning backbone.
  - `HF_TOKEN` — required only for gated target models (e.g. Llama).
- **Always run scripts from the repo root.** Paths to `.env`, `schema/`, and `outputs/` are resolved relative to the working directory (schema is found via `_resolve_schema`, which probes `./schema` and `../schema`).
- A CUDA GPU is recommended for local target models; scripts default to `device="cpu"`.

## Common Commands

```bash
# Main experiments (OpenAI GPT agent + local HF target model)
python test/test_qwen.py     # Qwen target
python test/test_llama.py    # Llama target
python test/test_ds.py       # DeepSeek target

# Ablation studies (adds a no-bridging "baseline" agent alongside PD-Agent)
python test_qwen_ablation.py
python test_llama_ablation.py

# Streamlit demo — LOCAL single-model path, no OpenAI key needed
./run_app.sh                 # or: streamlit run app.py
```

There is **no CLI / argparse**. Runtime settings are edited as constants at the **bottom of each script** (in the `__main__` block) and near the top:
- `TARGET_MODEL` (e.g. `"Qwen/Qwen3-1.7B"`), `num_questions` (interview turns, default 3), `device` (`"cpu"`/`"cuda"`) — bottom of each `test/*.py`.
- `TOOL_MODEL = 'gpt-4'` (the OpenAI reasoning backbone) — top of each script.

There is **no unit-test suite**: the files under `test/` are end-to-end *experiment* scripts, not pytest tests. Do not invoke a test runner; "running a test" here means executing one of these scripts.

## Architecture

There are **two distinct execution paths** that share the same schema and persona logic:

**1. Dual-agent path** (`test/test_*.py`, `test_*_ablation.py`) — the path used for paper results.
- **Tool Agent** = OpenAI `gpt-4`, wired via LangChain/LangGraph `create_agent` with a `MemorySaver` checkpointer and three tools: `create_bridging_extraction_prompt_tool`, `save_bridging_results_tool`, `create_and_save_graph_tool`.
- **Target LLM** = a local Hugging Face model wrapped by `HuggingFaceModelWrapper`, conditioned with a hidden persona system prompt.
- Orchestrated by `run_interview_system(...)`.

**2. Local single-model path** (`pipeline_local.py` → `run_local_pipeline`, consumed by `app.py`).
- Uses **one local Qwen model for every role** (interviewer, extractor, predictor) — no OpenAI dependency. This is what the Streamlit demo runs. `run_app.sh` hardcodes a specific interpreter path (`/opt/conda/envs/tracecal_rl/bin`) that has torch installed; adjust it for other machines.

**PD-Agent pipeline stages** (both paths follow the same logic):
1. **Persona assignment** — `create_target_persona()` samples a ground-truth persona from the schema; it is injected into the target's system prompt and hidden from the agent.
2. **Interactive interview** — `num_questions` turns; agent asks, target answers.
3. **Bridging extraction** — few-shot prompt (built from `schema/bridging_relationships.json`) extracts the 7 relation types into JSON → `outputs/bridging_results.json`.
4. **Graph construction** — relations become a directed semantic graph via `networkx` (`create_and_save_graph_tool`).
5. **Persona inference** — agent predicts the 4-dim persona from the graph.
6. **Scoring** — `QwenEmbeddingExtractor` + `compute_embedding_similarity` (`cosine_similarity`) compare prediction vs ground-truth → `outputs/persona_similarity.json`.

**Schema-driven** (`schema/`): `persona_schema.json` (4-dim structure + example values) and `bridging_relationships.json` (7 relation definitions/examples). Both paths load these rather than hardcoding categories.

**Persona-leak guard**: `_collect_persona_terms` + `redact_persona_leaks` strip explicit self-disclosures (e.g. "As a doctor…") from target responses before extraction, so the agent must *infer* rather than copy the persona.

**Outputs** (git-ignored `outputs/`): `interview_results_<model>.json`, `bridging_results.json`, graph structure, `persona_similarity.json`.

## Conventions & Gotchas

- **`test/test_qwen.py` is the reference implementation** (most complete, ~1100 lines). `test_llama.py`/`test_ds.py` and the two `*_ablation.py` scripts are **near-duplicates** with a different `TARGET_MODEL` and, for ablations, an added no-bridging baseline agent (`create_baseline_agent` / `run_baseline_ablation`). A change to shared logic must be **replicated across all of these files** — they do not import a common module.
- Scripts write into `outputs/` in the current directory, so running from a non-root directory scatters artifacts.
- `.DS_Store`, `__pycache__/`, `outputs/`, and `.env` are git-ignored.
- `소프트웨어등록.md` is a Korean software-registration document, not code.
