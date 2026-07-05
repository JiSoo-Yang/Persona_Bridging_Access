"""
PD-Agent Streamlit Demo (Qwen-only, no API key)
================================================
Interactive web UI for "The Pragmatic Persona: Discovering LLM Persona
through Bridging Inference" (ICPR 2026).

This demo runs FULLY LOCALLY with a single Qwen model — no OpenAI API key
required. The Qwen model plays every reasoning role (interviewer, bridging
extractor, persona predictor) as well as the target being interviewed.

Pipeline shown:
  random persona -> interview -> bridging extraction -> semantic graph
  -> persona prediction -> embedding cosine similarity

Run from the repository root (in an env with torch + transformers):
    streamlit run app.py
"""

import os
import io
import json
import importlib.util
from pathlib import Path
from contextlib import redirect_stdout

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)                       # outputs/ always land in code/outputs
OUTPUTS = REPO_ROOT / "outputs"

# Qwen model options (all local, none require an API key or gated access)
QWEN_MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

RUN_ARTIFACTS = [
    "bridging_results.json",
    "graph_structure.json",
    "graph_visualization.png",
    "persona_similarity.json",
]


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Import the local (API-key-free) pipeline once."""
    from pipeline_local import run_local_pipeline
    return run_local_pipeline


def clear_artifacts():
    for name in RUN_ARTIFACTS:
        p = OUTPUTS / name
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Page config & sidebar
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PD-Agent Demo (Qwen)", page_icon="🕵️", layout="wide")

st.title("🕵️ PD-Agent — Persona Discovery via Bridging Inference")
st.caption(
    "*The Pragmatic Persona: Discovering LLM Persona through Bridging Inference* (ICPR 2026) · "
    "**로컬 Qwen 전용 데모 (API 키 불필요)**"
)

# --- dependency guard: this Streamlit process must have torch installed ---
if importlib.util.find_spec("torch") is None:
    st.error(
        "현재 Streamlit이 실행된 파이썬 환경에 **torch**가 없습니다. "
        "이 앱은 torch·transformers가 설치된 환경에서 실행해야 합니다.\n\n"
        "저장소 루트에서 다음 중 하나로 실행하세요:\n"
        "```bash\n"
        "./run_app.sh\n"
        "# 또는\n"
        "/opt/conda/envs/tracecal_rl/bin/streamlit run app.py\n"
        "```"
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ 설정")

    model_id = st.selectbox("Qwen 모델", QWEN_MODELS, index=0)
    model_id = st.text_input("모델 ID (직접 입력 가능)", value=model_id)

    num_questions = st.slider("인터뷰 질문 수", min_value=1, max_value=8, value=3)
    device = st.radio("Device", ["cpu", "cuda"], index=0, horizontal=True)

    st.caption("⚠️ CPU + 대형 모델은 수 분 소요. 가장 가벼운 조합: Qwen3-0.6B / cpu.")

    st.divider()
    run = st.button("▶️  PD-Agent 실행", type="primary", use_container_width=True)

with st.expander("ℹ️ 파이프라인 개요", expanded=False):
    st.markdown(
        "단일 로컬 **Qwen** 모델이 인터뷰어·추출기·예측기·타겟 역할을 모두 수행합니다 (외부 API 없음).\n\n"
        "1. **랜덤 페르소나 생성·주입** — 4차원(Social Role·Personality·Background·Interests)\n"
        "2. **인터랙티브 인터뷰** — 질문 생성 + 답변의 페르소나 단서 마스킹\n"
        "3. **브리징 관계 추출** — 7종(part-of, member-of, instrument, theme, cause-of, in, temporal)\n"
        "4. **의미 그래프 구성·중심성 분석·시각화**\n"
        "5. **페르소나 예측 + 임베딩 코사인 유사도 평가**"
    )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def render_results(results: dict):
    gt = results.get("target_persona", {})
    ground_truth = results.get("ground_truth", {})
    conv = results.get("conversation_history", [])
    predicted = results.get("predicted_persona", {})
    sim = results.get("similarity_report", {})

    st.subheader("📈 페르소나 예측 유사도")
    if "scores" in sim:
        scores = sim["scores"]
        c = st.columns(5)
        c[0].metric("Social Role", f"{scores['social_role']:.3f}")
        c[1].metric("Personality", f"{scores['personality']:.3f}")
        c[2].metric("Background", f"{scores['background']:.3f}")
        c[3].metric("Interests", f"{scores['interests']:.3f}")
        c[4].metric("OVERALL", f"{scores['overall']:.3f}")
        st.bar_chart(
            {
                "similarity": {
                    "Social Role": scores["social_role"],
                    "Personality": scores["personality"],
                    "Background": scores["background"],
                    "Interests": scores["interests"],
                }
            }
        )
    else:
        st.warning(sim.get("error", "예측 페르소나 파싱 실패 — 유사도 계산 불가."))

    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.subheader("🎯 정답 페르소나 (Ground Truth)")
        st.json(ground_truth or gt)
    with col_pred:
        st.subheader("🤖 예측 페르소나 (Predicted)")
        st.json(predicted or {"(파싱 실패)": None})

    st.subheader("💬 인터뷰 (마스킹 적용)")
    for i, turn in enumerate(conv, 1):
        with st.chat_message("user"):
            st.markdown(f"**Q{i}.** {turn.get('question', '')}")
        with st.chat_message("assistant"):
            st.markdown(turn.get("answer", ""))

    st.subheader("🔗 추출된 브리징 관계")
    bpath = OUTPUTS / "bridging_results.json"
    if bpath.exists():
        rels = json.loads(bpath.read_text()).get("bridging_relations", [])
        if rels:
            st.dataframe(
                [
                    {
                        "anchor": r.get("anchor", ""),
                        "relation": r.get("relation_type", ""),
                        "anaphor": r.get("anaphor", ""),
                        "explanation": r.get("explanation", ""),
                    }
                    for r in rels
                ],
                use_container_width=True,
            )
        else:
            st.info("추출된 브리징 관계 없음.")
    else:
        st.info("브리징 결과 파일 없음.")

    st.subheader("🕸️ 의미 지식 그래프 (중심성)")
    gimg = OUTPUTS / "graph_visualization.png"
    if gimg.exists():
        st.image(str(gimg), use_container_width=True)
    else:
        st.info("그래프 이미지 없음 (추출된 관계가 없을 수 있음).")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    clear_artifacts()
    log_buffer = io.StringIO()
    status = st.empty()

    try:
        run_local_pipeline = get_pipeline()

        def progress(msg):
            status.info(msg)

        with st.spinner(f"'{model_id}' 로컬 실행 중… (CPU는 수 분 소요)"):
            with redirect_stdout(log_buffer):
                results = run_local_pipeline(
                    target_model_name=model_id,
                    num_questions=num_questions,
                    device=device,
                    progress=progress,
                )
        status.empty()
        st.success("완료.")
        render_results(results)
    except Exception as e:
        st.error(f"실행 오류: {e}")
        st.exception(e)
    finally:
        with st.expander("🖥️ 실행 로그 (stdout)", expanded=False):
            st.code(log_buffer.getvalue() or "(로그 없음)", language="text")
else:
    st.info("좌측 사이드바에서 모델을 고르고 **PD-Agent 실행**을 누르세요. API 키 불필요.")
