---
tags: [research, experiment, llm-persona, bridging-inference, ieee-access]
date: 2026-06-10
status: planned
---

# 실험 계획: PD-Agent IEEE Access 확장

## 0. 우선순위 (desk-reject 방어 관점)
- **P0 (없으면 desk reject)**: E1 평가 재설계 · E2 핵심 ablation · E3 강한 baseline → 비판①②③⑦ 직접 방어. **이 셋만으로 신규 분량 ~35% 충족.**
- **P1 (major revision 방어)**: E4 인간/LLM-judge · E5 robustness · E6 일반화.
- **P2 (스코프 결정)**: E7 steering(proof-of-concept).

## 1. 데이터셋
| 데이터셋 | 용도 | 주의 |
|---|---|---|
| PersonaChat / ConvAI2 | target conditioning + 자연 대화 GT | 문장형 persona → 4차원 라벨링 필요 |
| PersonaHub (Tencent) | 대규모·OOD·롱테일 직업 persona | 합성 noise 필터링 |
| Synthetic-Persona-Chat | 학습형 baseline(B3) 학습데이터 | |
| PANDORA (Reddit Big-Five/MBTI) | **Big-Five 외부 인간 GT** (현재 self-assign의 최약점) | 인간 라벨 신뢰도↑ |
| Essays (Pennebaker Big-Five) | Big-Five probe 검증 | |

**Ground-truth 3중 강화**: (1) behavioral validation — 주입 안 한 새 질문으로 manipulation check, 미발현 persona는 제외. (2) 인간 라벨 레이어(다중 평가자, Krippendorff α). (3) persona leak 정량 감사(GT 단어 직접 출현율, redaction 전후 성능차).

## 2. 베이스라인 (강화)
| ID | Baseline | 막는 비판 |
|---|---|---|
| B1 | Direct Probing ("describe your persona") | "그냥 물어보면?" |
| B2 | LLM-as-judge 직접 추론 (graph 없이, GPT-4o) | graph 순수 기여 분리 |
| B3 | Embedding probing (linear/logistic) | 학습형 baseline 부재 |
| B4 | Activation/internal probing (open-weight) | white-box 상한선 대조 |
| B5 | Full-dialogue CoT (강한 zero-shot) | straw-man 비판(⑦) |
| — | Vanilla, Frequency-Aware (기존) | lexical 하한 |

핵심 클레임: **PD-Agent > B1, B2, B5 가 유의하게** 성립해야 "구조 > 직접/lexical" 주장이 산다. B4 근접 시 "black-box인데 white-box probe에 근접" 셀링포인트.

## 3. 평가 메트릭 (E1 — cosine 강등)
| 차원 | 성질 | 주 메트릭 |
|---|---|---|
| Social Role | 범주 | Macro-F1, top-1/top-3 acc, 혼동행렬 |
| Big-Five | 연속/순위 | MAE, Spearman ρ, rank acc |
| Background | 다중속성 | per-attribute F1 (edu/loc/family) |
| Interests | 다중라벨 | Jaccard, multi-label F1 |
| (공통) | 신뢰도 | ECE + reliability diagram, chance/skyline 경계 |

**통계 규약(전 실험)**: persona N≥200/run, Wilcoxon signed-rank + BH 보정, paired bootstrap(10k) 95% CI, Cohen's d / Cliff's δ, backbone×target은 2-way mixed-effects.

## 4. Ablation Study 계획 (E2 — 빠지면 reject)
인과 사슬 `dialogue → relations → graph → centrality → persona` 마디별 절단:
- A1 graph 제거(relation list만) — graph 기여?
- A2 centrality 제거 / A3 centrality 교체(betweenness·eigenvector·PageRank·clustering)
- A4 **relation-type leave-one-out**(7개 각각) → 막대그래프+유의성 = 7-relation taxonomy 정당화
- A5 Association(2) vs Characterization(5) — 2-mode 유효성
- A6 인터뷰 턴 수 1/2/3/5/8/12(포화곡선) / A7 few-shot 0/1/3/5
- **A8 (비평가 핵심) relation 라벨 셔플** vs 정상 — bridging의 인과 기여 분리
- **A9 plain concept co-occurrence graph** vs bridging graph — bridging이 장식 아님 증명

## 5. Robustness / 일반화
- **E5a** 반복 추출 일관성: 동일 persona K=10회 → test-retest ICC, Fleiss κ (std<0.03 주장을 제대로 된 통계로 대체)
- **E5b** persona drift: 5→20턴 추출 변화 slope
- **E5c** 모호/충돌 persona: 모호도별 F1 + calibration
- **E5d** adversarial: "persona 숨겨라" 지시 시 F1 하락폭
- **E6** 외부 DS(PersonaChat/Hub) per-DS F1, OOD gap, 다국어(한/중) 언어별 F1(2-way ANOVA)

## 6. 인간 + LLM-judge (E4)
- 인간 ≥3명, ≥100 persona: forced-choice win-rate(sign test) + Likert 차원별(Krippendorff α, Fleiss κ).
- LLM-judge: target/임베더와 **다른 모델군**(GPT-4o judge), judge 2~3종 일치도.
- 인간 ↔ LLM-judge ↔ cosine 상관 → cosine 사용 정당화.

## 7. Steering 평가 (E7, P2 — open-weight 한정)
| 축 | 메트릭 | 검정 |
|---|---|---|
| 성공률 | target persona shift accuracy | McNemar |
| 정밀도 | off-target drift (의도 차원만 변화?) | per-dim Δ, paired t |
| 부작용 | MMLU/GSM8K 변화 | 비열등성 마진 |
| fluency | PPL, MAUVE | KS test |
| safety | toxicity rate | rate 비교 |
| 차별화 | naive CAA vs bridging-contrastive vs relation-selective | 3자 비교 |

## 8. 가장 시급한 단일 조치
> **메인 테이블의 cosine 단독 보고를 폐기**하고 차원별 F1 + bootstrap CI + Wilcoxon로 교체. 0.99는 강점이 아니라 reviewer에게 "평가가 깨졌다"는 신호.

## 9. 예상 일정 (대략)
1. (2~3주) E1 프로토콜 재설계 + 데이터 파이프라인 + leak audit
2. (3~4주) E2 ablation 전체 + E3 baseline
3. (2주) E4 인간/LLM-judge, E5 일관성
4. (3주) M1 BIG-Net 구현·학습 + M2 PCI
5. (3~4주, 선택) M3 B1 정렬분석 + B2 steering PoC (open-weight)
6. (2주) 작성·통계·figure

## Related Notes
- [[2026-06-10_PersonaBridging_토론로그]]
- [[2026-06-10_PersonaBridging_방법론]]
