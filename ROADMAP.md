# ROADMAP.md — IEEE Access 확장

이 문서는 PD-Agent(ICPR 2026)를 IEEE Access 저널로 확장하기 위한 **작업 목록**이다.
(CLAUDE.md = 리포에서 일하는 법 / ROADMAP.md = 지금 무엇을 할지.
세부 근거는 리포의 `research_notes/`(방법론·실험계획·토론로그) 참조.)

## 한 줄 방향 (single message)
> **LLM persona는 표면 어휘가 아니라 담화 결합(bridging) 구조에 *인과적으로* 인코딩된다.
> 같은 표상으로 persona를 추출(forward)하고 조종(inverse)할 수 있으며, 둘의 일치가 인과성의 증거다.**

## 제약 (반드시 지킬 것)
- IEEE Access = 컨퍼런스 대비 **신규 기여 30~40%**, 유사도 35% 미만.
- 저널 기여의 핵심은 "새 기능"이 아니라 **기존 주장을 검증하는 ablation + 정직한 평가**.
- 현재 cosine 0.90~0.99는 **천장 효과**라 정확도로는 못 판다 → 가치를 해석가능성·인과·진단으로.
- **범위 고정**: multilingual / multimodal 확장은 이번 논문에서 제외(메시지 분산 방지).

---

## Workstream A — 평가·Ablation (P0, 최우선 · 없으면 desk reject)
현재 `run_interview_system` / `compute_embedding_similarity`의 평가를 재설계한다.

- [ ] **E1 평가 프로토콜 재설계**: cosine 단독 → 차원별. Social Role=Macro-F1,
      Big-Five=MAE+Spearman ρ, Background=per-attribute acc, Interests=multi-label F1.
      + ECE(calibration), chance/skyline 경계, Wilcoxon+BH보정+paired bootstrap CI.
- [ ] **E2 핵심 ablation** (인과 사슬 절단): graph 제거 / centrality 제거·교체(betweenness·PageRank) /
      relation leave-one-out(7종) / Association vs Characterization / 턴 수 / few-shot 수 /
      **relation 유형 무작위 재샘플링**(A8) / **plain co-occurrence graph**(A9).
- [ ] **E3 강한 baseline 추가**: Direct Probing("describe your persona"), LLM-as-judge 직접추론,
      embedding probing, full-dialogue CoT. (현재 Vanilla/Frequency는 너무 약함.)

## Workstream B — BIG-Net (M1, "추출 심화")
degree centrality(≈빈도)를 학습된 graph representation으로 교체.

- [ ] 프로토타입은 리포의 `bignet_prototype/`(persona/datagen/features/model/train)에 있음.
      이 리포의 실제 파이프라인과 연결(아래 datagen 교체).
- [ ] `datagen.py`(합성) → **실제 추출 결과 로더**로 교체:
      `outputs/bridging_results.json` + ground-truth persona → `BridgingGraph`.
- [ ] R-GCN 학습 파이프라인을 실제 데이터로 검증(현재는 forward 수학 + 합성 학습가능성만 검증됨).
- [ ] **해석 figure**: 학습된 `W_r` norm / attention → "관계 → persona 차원" 매핑 시각화.
- [ ] motif로 Association/Characterization 2-mode 정량화.

## Workstream C — 진단 지표 (M2, 보완재 · 저위험)
- [ ] **PCI(Persona Consistency Index)**: 턴별 누적 그래프의 분포·구조·허브 안정성.
- [ ] **drift / hallucination 탐지**: dangling/disconnected component = bridging anchor 실패.
- [ ] 검증: adversarial 인터뷰 vs 정상에서 PCI/drift AUROC.

## Workstream D — 인과 검증 (M3, open-weight 한정 · proof-of-concept)
> steering은 **별도 기여가 아니라 추출이 인과적임을 증명하는 도구**로만.

- [ ] **B1 정렬 분석(인과 다리, 필수)**: open target(Llama/Qwen)에서 persona vector
      (Chen et al. 2025, arXiv 2507.21509) 추출 → BIG-Net graph 신호와 상관.
- [ ] **B2 bridging-guided steering(PoC)**: graph 기반 contrastive pair로 steering 벡터 유도 →
      재추출로 controllability 확인. 차별화 3자 ablation(naive CAA vs bridging vs relation-selective).
- [ ] 부작용 평가: off-target drift, fluency(PPL/MAUVE), safety.

## Workstream E — 일반화·인간평가 (P1)
- [ ] **E4** 인간평가 + LLM-as-judge(독립 모델군, Krippendorff α).
- [ ] **E5** robustness: 반복 추출 ICC, persona drift, 모호/적대 persona.
- [ ] **E6** 외부 데이터셋(PersonaChat/PersonaHub/PANDORA), OOD persona.
- [ ] ground-truth 3중 강화: behavioral validation, 인간 라벨, persona leak 정량 감사.

---

## 우선순위 요약
1. **P0**: Workstream A(E1·E2·E3) — 이것만으로 신규 분량 ~35%.
2. **P1**: Workstream B(BIG-Net) + E(일반화/인간평가).
3. **P2**: Workstream C(진단), D(인과검증 PoC).

## 다음 액션 (추천 시작점)
`test/test_qwen.py`의 `compute_embedding_similarity`를 차원별 metric으로 확장(E1) →
같은 하네스에서 relation-shuffle ablation(E2/A8) 추가. 여기부터 시작하면 리스크가 가장 낮다.
