---
tags: [research, debate, llm-persona, bridging-inference, ieee-access]
date: 2026-06-10
status: discussed
---

# 연구 토론: PD-Agent의 IEEE Access 확장 방향

## 핵심 질문 (Research Question)
PD-Agent가 bridging inference로 target LLM의 persona를 *추출(관찰자 입장)* 하는 현재 방향을, 저널급(IEEE Access, 신규기여 30~40%) 기여가 되도록 어떻게 발전시킬 것인가. 특히 activation steering은 "추출 입장"과 결이 다른데 어떻게 다뤄야 하나.

## 참여 전문가
- 📚 문헌조사 전문가 (Literature Surveyor)
- 🔧 방법론 전문가 (Methodology Architect)
- 🧪 실험설계 전문가 (Experiment Designer)
- 🎯 비평가 (Critical Reviewer)

---

## Phase 2: 개별 분석 요약

### 📚 문헌조사 분석
- **연구 지형(2024~2025)**: persona 연구가 `부여(assignment) → 행동 추론(inference) → 내부 표현 추출(extraction)`로 이동. PD-Agent는 "관찰자(black-box) × 구조적(discourse) 추론"의 교집합에 위치 — 이 자리는 비어 있어 유리.
- **가장 가까운 선행연구**: `Tang et al. 2025, ActivePersona`(응답 생성 전 persona 추론). 차별화 문장 필수.
- **반드시 인용/차별화해야 할 영역**: **Persona Vectors** (`Chen, Arditi, Sleight, Evans, Lindsey, arXiv 2507.21509, 2025`) — trait 유발/억제 프롬프트의 activation 차이를 persona 방향으로. RepE(Zou 2023), CAA(Panickssery/Rimsky 2023), ITI(Li 2024), ActAdd(Turner 2023).
- **차별화 무기**: persona vectors는 **white-box + 단일 trait 방향**, PD-Agent는 **black-box + 다차원 schema + 대화 행동 기반**. → closed API large model에도 적용 가능.
- **이론적 빈자리(가장 큼)**: 기존 bridging 연구는 모두 "텍스트 자체의 coherence resolution"(Hou 2013; SOTA F<0.5로 어려운 과제). **"화자의 잠재 정체성을 드러내는 신호로서의 bridging"은 거의 다뤄진 적 없음** → PD-Agent의 진짜 novelty 자리.
- **신규 검증 자원**: `GUMBridge (Levine & Zeldes 2025)`, ARRAU 등 bridging-annotated corpus로 추출 정확도를 외부 검증 가능(현재 원고엔 추출 검증이 전무).

### 🔧 방법론 제안
현재 코어의 약점: **degree centrality는 graph를 "쓰는" 게 아니라 빈도 요약 수준**. graph가 추론 엔진이 아니라 시각화용. → 여기가 최대 기회.
- **제안 1 (메인, 저위험) — BIG-Net**: degree centrality 대신 bridging graph를 **R-GCN/RGAT로 graph-level 임베딩** → 4차원 persona를 직접 회귀/분류. relation-type별 가중치 W_r를 학습해 "어떤 관계가 어떤 차원을 결정하는가" 해석. **motif(3-node)** 로 Association/Characterization 2-mode를 정량화(이론 정합). 가치는 정확도가 아니라 **해석가능성·추론효율(LLM 호출 없이)·zero-shot transfer**.
- **제안 2 (고위험·고임팩트) — Bridging-Guided Steering**: graph가 정의하는 행동 차이(특정 relation 유발 여부)에서 contrastive하게 steering 벡터 유도. 일반 CAA와 달리 추출 프레임워크와 **동일 이론 축**에서 닫힘. 단, "graph 유도 ≈ 단순 prompt-pair 벡터"일 위험 → 반드시 3자 ablation. 메인 아닌 **보조 1섹션** 권장.
- **제안 3 (보완재, 저위험) — PCI/Drift**: 턴별 누적 temporal graph로 **Persona Consistency Index**와 drift/hallucination 탐지. hallucination = "bridging anchor 실패한 dangling 노드"로 이론적 정의(SDRT weak referent와 정합). 학습 불필요, networkx만으로 빠르게.
- **추천 조합**: **제안1(메인) + 제안3(보완)** 확정 + 제안2는 예비실험.

### 🧪 실험설계 계획
- **현재 평가가 구조적으로 깨짐**: cosine 0.90~0.99는 ceiling effect, ground-truth가 self-assigned(순환), 임베더(Qwen3-1.7B)가 target이기도 함(self-referential), 통계검정 전무.
- **P0 (없으면 desk reject)**:
  - **E1 평가 프로토콜 재설계**: cosine 강등. 차원별 분리 — Social Role→Macro-F1, Big-Five→MAE/Spearman ρ, Background→per-attribute F1, Interests→multi-label F1. + ECE(calibration) + chance/skyline 경계. + Wilcoxon, paired bootstrap CI, Cohen's d, BH 보정.
  - **E2 핵심 ablation**: graph 제거 / centrality 제거 / centrality 교체(betweenness·PageRank) / relation leave-one-out / Association vs Characterization / 턴 수 / few-shot 수.
  - **E3 강한 baseline**: Direct Probing("describe your persona"), LLM-as-judge 직접추론, embedding probing, activation probing(open-weight 상한선).
- **P1**: E4 인간평가+LLM-judge(독립 모델군, Krippendorff α), E5 robustness(test-retest ICC, drift, 모호/적대 persona), E6 일반화(PersonaChat/PersonaHub, OOD, 다국어).
- **데이터셋**: PersonaChat/ConvAI2, PersonaHub, PANDORA(Big-Five 외부 GT). **ground-truth 3중 강화**: behavioral validation(주입 안 한 질문으로 manipulation check), 인간 라벨 레이어, persona leak 정량 감사.
- **steering 확장 시(E7)**: 성공률만으론 부족 — off-target drift, fluency(PPL/MAUVE), safety 부작용 3종 필수.

### 🎯 비평적 분석 (가장 날카로움)
- **비판①(치명)**: bridging의 인과 기여가 분리 안 됨. 0.77→0.91 향상이 bridging 때문인지 그냥 "구조화 CoT + 더 긴 추론" 때문인지 구별 불가. → relation 라벨 셔플 ablation, plain co-occurrence graph ablation 필수.
- **비판②**: degree centrality ≈ 빈도. edge label이 degree 계산에 안 들어감 → 7 relation 구분 이유 소멸. "graph가 아무것도 안 한다."
- **비판③**: 순환 평가 + cosine 0.99 무의미(short text cosine floor 높음). chance-level cosine 미보고가 결정타. 임베더가 target과 같은 모델군.
- **비판④(의인화)**: 제목 "Discovering"인데 실제론 "주입한 라벨 회수(recovering injected label)". default persona 추출 실험 없으면 overclaim.
- **비판⑤**: Irmer 7관계를 LLM에 강제하면 LLM은 뭐든 7개로 우겨넣음 → 추출 신뢰도(human κ) 검증 필요.
- **비판⑥⑦**: 너무 매끈한 결과(재현성/시드/버전), straw-man baseline(강한 full-dialogue CoT 부재).

---

## Phase 3: 교차 토론 — 수렴

### ✅ 합의된 사항 (4인 독립 수렴)
1. **저널 기여의 70%는 "새 capability"가 아니라 "기존 주장을 죽이려는 ablation + 정직한 평가"다.** (방법론·실험·비평 모두 동의) → bridging의 인과 기여 분리(E2) + 순환평가 해소(E1) 없이는 무엇을 붙여도 토대가 무너짐.
2. **graph를 진짜 추론 엔진으로 승격**해야 함. degree centrality(=빈도)를 학습된 graph representation(GNN/motif)으로 교체 → "왜 graph인가"에 답.
3. **steering에 대한 사용자 직관은 절반만 맞다.** 별도 기여로 팔면 "두 논문을 본드로 붙인" 프랑켄슈타인 + persona-vector 문헌 대비 새로움 약함 → 위험. **그러나** 추출↔개입을 forward/inverse로 묶으면 steering은 **추출이 인과적임을 증명하는 검증 도구**가 됨(비판①·④ 동시 방어). open-weight 모델로 한정.
4. **persona vectors(2507.21509)와의 black-box vs white-box 대비**가 positioning의 핵심 무기.
5. **단 하나의 메시지로 수렴**: *"LLM persona는 표면 어휘가 아니라 담화 결합(bridging) 구조에 **인과적으로** 인코딩되어 있으며, 같은 표상으로 추출(forward)하고 조종(inverse)할 수 있다 — 상관이 아니라 인과다."*

### ⚠️ 남은 쟁점
- **steering을 넣을 것인가?** 비평가는 "인과 검증 루프로 종속시키면 강력", 방법론가는 "고위험이라 예비실험으로". → **결정: open-weight(Llama/Qwen) 한정 proof-of-concept 1섹션 + B1(graph 신호 ↔ persona vector 정렬) 상관분석을 인과 다리로.** 전면에 걸지 않음.
- **메인 기여를 GNN(BIG-Net)으로 할지, 인과검증(steering loop)으로 할지**: 둘은 양립 가능 — GNN = "추출 심화", steering = "인과 검증", 정직한 평가/ablation = "토대". 분량·리스크상 GNN+정직평가를 본체로, steering을 검증 절로.
- multilingual/multimodal(기존 future work)은 **이번 확장에서 제외** — 메시지 분산 방지.

## 인용 정확도 주의
- 확실: Persona Vectors(Chen et al. 2507.21509, 2025), RepE/CAA/ITI/ActAdd, GUMBridge(2025), 원고 내 인용.
- 확인 필요(검색 스니펫 기반): BILLY/PERSONA/Style Modulation Heads 서지, AI PERSONA·PICLe, "Localizing Persona Representations"(AIES 2025) 저자명 — 본문 인용 전 arXiv abstract로 1차 확인.

## Related Notes
- [[2026-06-10_PersonaBridging_방법론]]
- [[2026-06-10_PersonaBridging_실험계획]]
