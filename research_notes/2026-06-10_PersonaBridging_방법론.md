---
tags: [research, methodology, llm-persona, bridging-inference, ieee-access]
date: 2026-06-10
status: proposed
---

# 제안된 방법론: Causal Bridging Persona (PD-Agent 저널 확장)

## 1. Problem Statement
현재 PD-Agent는 `대화 → bridging relation 추출 → directed graph → degree centrality → LLM이 persona 텍스트 추론 → cosine 평가`. 두 가지 근본 한계:
1. **graph가 추론 엔진이 아님** — degree centrality는 사실상 개념 빈도 요약이고, 7개 relation 라벨은 centrality 계산에 들어가지도 않음. "왜 graph/bridging인가"가 증명 안 됨.
2. **추출이 인과가 아니라 상관** — "bridging 패턴이 persona를 *드러낸다*"는 관찰만 있고, 그 표상이 모델 내부에 *인과적으로* 자리한다는 증거가 없음.

## 2. Proposed Method

### 2.1 Overview — 하나의 메시지
> **LLM persona는 담화 결합(bridging) 구조에 인과적으로 인코딩된다. 같은 bridging 표상으로 persona를 추출(forward)하고 조종(inverse)할 수 있으며, 둘의 일치가 인과성의 증거다.**

세 모듈로 구성:
- **(M1) 추출 심화**: degree centrality → 학습된 graph-level representation(BIG-Net) + motif.
- **(M2) 진단**: temporal bridging graph로 Persona Consistency Index(PCI) & drift/hallucination 탐지.
- **(M3) 인과 검증**: bridging 신호 ↔ 내부 persona 방향(activation) 정렬 + bridging-guided steering(open-weight 한정, proof-of-concept).

### 2.2 M1 — BIG-Net (Bridging-Inference Graph Network)

```
Input: R = {(anchor a_i, anaphor b_i, relation_type t_i, rationale s_i)}

[Stage 1] Featurization
  node x_v = SentenceEmbed(concept_v)          # all-MiniLM-L6-v2 또는 독립 임베더(E5/text-embedding-3)
  edge e_(u→v) = OneHot(t_i) ⊕ Embed(rationale) # directed typed multigraph

[Stage 2] Relational GNN (relation-type별 분리 가중치)
  h_v^(l) = σ( Σ_{r∈7} Σ_{u∈N_r(v)} (1/c_{v,r}) W_r^(l) h_u^(l-1) + W_0^(l) h_v^(l-1) )
  # W_r 가 "어떤 relation이 어떤 persona 차원을 결정하는가" 학습 → 해석 figure

[Stage 3] Graph-level readout
  z_G = AttentionPool({h_v^(L)}) ⊕ RelationHistogram(R) ⊕ MotifVector(G)
  # MotifVector: 3-node motif 빈도. part-of→member-of(분해자) vs cause-of→temporal(서사자)
  #              → Irmer의 Association/Characterization 2-mode를 정량화 (이론 정합)

[Stage 4] Multi-task persona head
  ŷ_role  = softmax(MLP_role(z_G))     # 범주
  ŷ_big5  = sigmoid(MLP_big5(z_G))     # 5-dim 연속
  ŷ_bg, ŷ_int 동일
  Loss = Σ_dim CE/MSE
```
- **스택**: PyTorch Geometric(`RGCNConv`/`RGATConv`, attention pooling), networkx `triadic_census`(motif).
- **데이터 병목 없음**: persona를 직접 부여하므로 labeled graph 무제한 합성.
- **가치 재정의(중요)**: cosine은 이미 천장 → BIG-Net의 가치는 정확도가 아니라 **(a) 해석가능성(W_r, motif-persona 매핑 figure), (b) 추론 효율(LLM 호출 없이 graph만으로 추론), (c) 미학습 target에 zero-shot transfer.**

### 2.3 M2 — Persona Consistency Index (PCI) & Drift
```
턴 t까지 누적 graph G_t, relation 분포 r_t (7-dim)
PCI = w1·(1 − mean_t JSD(r_t‖r_{t-1}))            # 분포 안정성
    + w2·mean_t GraphSim(G_t,G_{t-1})              # 구조 안정성(motif-spectrum cosine/GED)
    + w3·KendallTau(hub_t, hub_{t-1})              # 허브 일관성
drift δ_t = JSD(r_t‖r_global) + λ·neighborhood_novelty
hallucination 단서 = dangling/disconnected component 급증 (= bridging anchor 실패)
```
- **이론 정합**: bridging의 본질이 "새 개념을 기존 frame의 weak referent에 anchor". anchor 실패 노드 = frame 밖 환각. SDRT/weak referent 이론에서 직접 도출.
- **스택**: networkx + scipy. 학습 불필요 → 빠른 결과(risk hedge).

### 2.4 M3 — 인과 검증 (추출 ↔ 제어 루프, open-weight 한정)
- **B1 정렬 분석(인과 다리, 필수)**: Llama/Qwen target에서 persona vector(Chen et al. 2507.21509 방식: trait 유발/억제 프롬프트 activation 차이) 추출 → **PD-Agent가 추론한 graph 신호(relation density 등) ↔ internal vector 투영값의 상관**. 유의한 상관 = "행동적 bridging이 내부 persona 방향과 정렬" = 추출이 표면 아티팩트 아님.
- **B2 bridging-guided steering(proof-of-concept)**: graph가 정의하는 contrastive pair(특정 relation 유발 발화 D+ vs D-)에서 diff-in-means로 steering 벡터 → 적용 후 **PD-Agent로 재추출**해 목표 방향 이동 확인(controllability matrix).
  - **차별화 입증 필수**: (a) naive prompt-pair CAA vs (b) bridging-contrastive vs (c) relation-selective 3자 ablation 없으면 "CAA 재탕" 비판.
  - **스택**: TransformerLens(hook), `repeng`/steering-vectors. proprietary backbone은 activation 불가 → open-weight만, 솔직히 명시.

## 3. 예상 기여점 (Expected Contributions)
1. (이론) bridging resolution을 **화자 정체성의 진단 신호**로 재해석 — 문헌상 빈자리.
2. (방법) degree centrality → **학습된 graph representation + motif**로 persona를 graph-level로 직접 인코딩(BIG-Net).
3. (인과) bridging 신호 ↔ 내부 persona 방향 정렬 + steering 검증으로 **상관을 인과로** 승격.
4. (평가) per-dimension·chance-level·독립 임베더·인간/LLM-judge·통계검정으로 평가 정직성 확보.
5. (positioning) **black-box 다차원 vs white-box 단일방향(persona vectors)** 명확화.

## 4. 예상 한계점 및 대응
- BIG-Net이 정확도로는 LLM 추론을 못 이길 수 있음 → 가치를 해석가능성/효율/transfer로 프레이밍.
- steering 차별성 입증 실패 위험 → 3자 ablation + 메인 아닌 검증 절로 종속, open-weight 한정.
- 추출 신뢰도 미검증 → GUMBridge/ARRAU 외부 corpus로 추출 F-score 보고.
- 스코프 폭증 → multilingual/multimodal 제외, steering은 PoC.

## Related Notes
- [[2026-06-10_PersonaBridging_토론로그]]
- [[2026-06-10_PersonaBridging_실험계획]]
