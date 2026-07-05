"""
Local, API-key-free PD-Agent pipeline (Qwen-only)
=================================================
A self-contained variant of the PD-Agent pipeline that uses a single **local
Qwen** model for every reasoning role (interviewer, bridging extractor, persona
predictor) instead of the OpenAI GPT tool-agent. No external API key required.

It reuses the pure helper functions from ``test/test_qwen.py`` (persona
generation, persona redaction, bridging-prompt construction, graph
construction/visualization, embedding cosine similarity) so the underlying
method stays identical to the paper implementation — only the orchestration
backbone is swapped from a remote GPT agent to the local model.

Public entry point:
    run_local_pipeline(target_model_name, num_questions, device, progress=None)
returns a results dict with the same shape as
``test/test_qwen.py:run_interview_system``.
"""

import os
import re
import json
import time
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- import the pure helpers from test/test_qwen.py (no LangChain needed) ---
_spec = importlib.util.spec_from_file_location(
    "pdagent_qwen_helpers", REPO_ROOT / "test" / "test_qwen.py"
)
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)


class _Msg:
    """Minimal message object understood by HuggingFaceModelWrapper._to_dict."""

    def __init__(self, role: str, content: str):
        self.type = role          # "system" | "user" | "assistant"
        self.content = content


def _build_persona_system_msg(persona: dict) -> str:
    """Persona role-play system prompt (mirrors create_target_llm)."""
    return f"""You are roleplaying as a person with the following characteristics:

Social Role: {persona.get('social_role', 'unknown')}
Personality Traits: {json.dumps(persona.get('personality', {}), indent=2)}
Background: {json.dumps(persona.get('background', {}), indent=2)}
Interests: {json.dumps(persona.get('interests', {}), indent=2)}

STRICT OUTPUT RULES:
- Answer in exactly ONE complete sentence.
- Do NOT include any XML/angle-bracket tags like <think>, <analysis>, or similar.
- Do NOT reveal internal thoughts, step-by-step reasoning, or meta comments.
- Do NOT start with "As a ..." or explicitly state your job title.
- If unsure, just provide your best single-sentence reply without showing any reasoning."""


def _parse_bridging_json(text: str) -> dict:
    """Robustly pull a {"bridging_relations": [...]} object out of model text."""
    if not text:
        return {"bridging_relations": []}

    candidates = []
    candidates += re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)

    # balanced-brace scan for top-level {...} objects
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for cand in reversed(candidates):
        try:
            obj = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "bridging_relations" in obj:
            return obj
    return {"bridging_relations": []}


def _normalize_persona(obj: dict, template: dict) -> dict:
    """Coerce a loosely-parsed dict into the canonical 4-dimension persona shape."""
    def _s(x):
        return x if isinstance(x, str) else ("" if x is None else str(x))

    pers = obj.get("personality") if isinstance(obj.get("personality"), dict) else {}
    bg = obj.get("background") if isinstance(obj.get("background"), dict) else {}
    it = obj.get("interests") if isinstance(obj.get("interests"), dict) else {}
    t_pers = template.get("personality", {})
    t_bg = template.get("background", {})
    t_it = template.get("interests", {})
    return {
        "social_role": _s(obj.get("social_role", "")),
        "personality": {
            "trait": _s(pers.get("trait", t_pers.get("trait", "unknown"))),
            "value": _s(pers.get("value", "")),
        },
        "background": {
            "category": _s(bg.get("category", t_bg.get("category", "unknown"))),
            "value": _s(bg.get("value", "")),
        },
        "interests": {
            "category": _s(it.get("category", t_it.get("category", "unknown"))),
            "value": _s(it.get("value", "")),
        },
    }


def _extract_persona_lenient(text: str, template: dict):
    """Parse a persona dict from model text; accepts partial objects."""
    if not text:
        return None
    candidates = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)

    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    keys = ("social_role", "personality", "background", "interests")
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in keys):
            norm = _normalize_persona(obj, template)
            if (
                norm["social_role"]
                or norm["personality"]["value"]
                or norm["background"]["value"]
                or norm["interests"]["value"]
            ):
                return norm
    return None


def run_local_pipeline(
    target_model_name: str = "Qwen/Qwen3-1.7B",
    num_questions: int = 3,
    device: str = "cpu",
    progress=None,
) -> dict:
    """Run the full PD-Agent pipeline locally with a single Qwen model."""

    def _say(msg: str):
        print(msg)
        if progress:
            progress(msg)

    os.makedirs("outputs", exist_ok=True)

    # 1) Random ground-truth persona -------------------------------------------------
    _say("① 랜덤 페르소나 생성 중…")
    persona, ground_truth, prediction_template = Q.create_target_persona()
    persona_terms = Q._collect_persona_terms(persona, ground_truth)
    target_system_msg = _build_persona_system_msg(persona)

    # 2) Load ONE local Qwen model for all reasoning roles ---------------------------
    _say(f"② 로컬 모델 로딩: {target_model_name} …")
    model = Q.HuggingFaceModelWrapper(target_model_name, device=device, max_new_tokens=512)

    interviewer_sys = _Msg(
        "system",
        "You are an interviewer trying to discover a person's hidden persona "
        "(their job, personality, background, interests). Ask ONE short, natural "
        "question at a time. Output ONLY the question text, nothing else.",
    )

    # 3) Interactive interview -------------------------------------------------------
    conversation_history = []
    target_conversation = [_Msg("system", target_system_msg)]

    for i in range(num_questions):
        _say(f"③ 인터뷰 {i + 1}/{num_questions} …")
        if i == 0:
            user = _Msg("user", "Ask your first question to get to know the person.")
        else:
            hist = "\n".join(
                f"Q: {t['question']}\nA: {t['answer']}" for t in conversation_history
            )
            user = _Msg(
                "user",
                f"Conversation so far:\n{hist}\n\nAsk your next question "
                f"({i + 1}/{num_questions}). Output only the question.",
            )
        question = Q.clean_agent_question(model.invoke([interviewer_sys, user]).content)

        target_conversation.append(_Msg("user", question))
        answer_raw = model.invoke(target_conversation).content
        answer = Q.redact_persona_leaks(answer_raw, persona_terms)
        target_conversation.append(_Msg("assistant", answer))
        conversation_history.append({"question": question, "answer": answer})

    conversation_json = json.dumps(conversation_history, ensure_ascii=False, indent=2)

    # 4) Bridging relation extraction ------------------------------------------------
    _say("④ 브리징 관계 추출 중…")
    extraction_prompt = Q.create_bridging_extraction_prompt_tool(conversation_json)
    extraction_msgs = [
        _Msg("system", "You are an expert in bridging inference. Output ONLY valid JSON."),
        _Msg("user", extraction_prompt),
    ]
    bridging = _parse_bridging_json(model.invoke(extraction_msgs).content)
    bridging_json = json.dumps(bridging, ensure_ascii=False)
    Q.save_bridging_results_tool(bridging_json, "bridging_results.json")

    # 5) Semantic graph construction + visualization ---------------------------------
    _say("⑤ 의미 그래프 구성·시각화 중…")
    Q.create_and_save_graph_tool(bridging_json, "graph")
    graph_summary = {}
    gpath = REPO_ROOT / "outputs" / "graph_structure.json"
    if gpath.exists():
        graph_summary = json.loads(gpath.read_text()).get("summary", {})

    # 6) Persona prediction ----------------------------------------------------------
    _say("⑥ 페르소나 예측 중…")
    prediction_prompt = f"""Based on the interview and the bridging-relation graph, predict the person's persona.

PREDICTION TARGETS (predict the VALUES; categories are given):
{json.dumps(prediction_template, ensure_ascii=False, indent=2)}

Interview:
{conversation_json}

Bridging graph summary:
{json.dumps(graph_summary, ensure_ascii=False)}

Return ONLY a single JSON object in this EXACT format (no other text, no reasoning, no explanation).
Start your reply with '{{' and end with '}}':
{{
  "social_role": "predicted role",
  "personality": {{ "trait": "{prediction_template.get('personality', {}).get('trait', 'unknown')}", "value": "yes or no" }},
  "background": {{ "category": "{prediction_template.get('background', {}).get('category', 'unknown')}", "value": "predicted value" }},
  "interests": {{ "category": "{prediction_template.get('interests', {}).get('category', 'unknown')}", "value": "predicted value" }}
}}"""
    prediction_msgs = [
        _Msg("system", "You output ONLY a JSON object and nothing else. No reasoning, no prose."),
        _Msg("user", prediction_prompt),
    ]

    # Try greedy first, then retry with sampling if the JSON cannot be parsed.
    predicted_persona = {}
    final_text = ""
    for attempt in range(3):
        final_text = model.invoke(
            prediction_msgs, do_sample=(attempt > 0), temperature=0.7
        ).content
        cand = _extract_persona_lenient(final_text, prediction_template)
        if cand:
            predicted_persona = cand
            break
        _say(f"⑥ 예측 JSON 파싱 재시도 {attempt + 1}/3 …")

    # 7) Embedding cosine similarity -------------------------------------------------
    if predicted_persona:
        _say("⑦ 임베딩 유사도 계산 중…")
        embedder = Q.QwenEmbeddingExtractor(target_model_name)
        similarity_report = Q.compute_embedding_similarity(
            ground_truth, predicted_persona, embedder
        )
    else:
        similarity_report = {
            "error": "Could not parse predicted persona JSON from the local model output.",
            "ground_truth": ground_truth,
            "predicted": predicted_persona,
        }
    Q.save_json(similarity_report, "outputs/persona_similarity.json")

    results = {
        "target_model": target_model_name,
        "target_persona": persona,
        "ground_truth": ground_truth,
        "prediction_template": prediction_template,
        "conversation_history": conversation_history,
        "agent_final_text": final_text,
        "predicted_persona": predicted_persona,
        "similarity_report": similarity_report,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Q.save_json(
        results, f"outputs/interview_results_{target_model_name.replace('/', '_')}.json"
    )
    _say("완료 ✅")
    return results
