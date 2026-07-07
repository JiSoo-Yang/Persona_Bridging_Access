# BIG-Net Prototype — Bridging-Inference Graph Network

PD-Agent의 저널 확장(M1: "추출 심화")을 위한 프로토타입.
현재 논문의 **degree centrality(사실상 개념 빈도)** 를, bridging graph를 직접 학습하는
**graph-level 표현(BIG-Net)** 으로 교체한다.

---

## 1. BIG-Net이 뭔가? (처음 보는 사람용 5분 설명)

### 문제
현재 PD-Agent는 대화에서 7가지 bridging 관계를 뽑아 그래프 `G=(V,E)`를 만든 뒤,
**노드의 degree centrality** 하나로 persona를 추론한다. 그런데 degree는
"이 개념이 몇 번 연결됐나"라는 **빈도**에 가깝고, **엣지의 관계 유형(part-of/cause-of…)은
아예 계산에 안 들어간다.** → "왜 굳이 graph인가, 7개 관계를 왜 구분했나"에 답을 못 한다.

### 아이디어
그래프를 **신경망으로 직접 읽어** persona를 예측하자. 이때 핵심은 **관계 유형별로
다르게 메시지를 전달**하는 것. 이 역할을 하는 게 **R-GCN(Relational Graph Convolutional
Network)** 이다.

### GNN / R-GCN 직관
- **GNN(그래프 신경망)**: 각 노드가 "이웃 노드들의 정보를 모아 자기 벡터를 갱신"하는 과정을
  여러 겹 반복한다. 몇 겹 지나면 각 노드 벡터가 "그래프 안에서 내 주변 구조"를 요약하게 된다.
- 보통 GCN은 이웃을 **똑같은 가중치**로 섞는다.
- **R-GCN**은 엣지의 **관계 유형 r마다 별도 가중치 행렬 `W_r`** 를 둔다. 즉
  `part-of`로 연결된 이웃과 `cause-of`로 연결된 이웃을 **다른 방식으로** 받아들인다.
  → bridging 관계 유형이 실제로 계산에 반영된다(현재 degree centrality의 결정적 약점 해결).

  노드 v의 갱신식:
  ```
  h_v ← ReLU( W_0 h_v  +  Σ_r Σ_{u ∈ N_r(v)}  (1/c_{v})  W_r h_u )
  ```
  (`W_0`=자기 자신, `W_r`=관계 r 이웃, `c_v`=정규화용 in-degree)

### 전체 파이프라인 (4단계)
| 단계 | 하는 일 | 코드 |
|---|---|---|
| 1. Node embedding | concept id → 벡터 (`nn.Embedding`; 실제로는 문장 임베더로 교체) | `model.py: BIGNet.emb` |
| 2. Relational GNN | R-GCN 레이어 L개로 노드 벡터 갱신 | `model.py: RGCNLayer` |
| 3. Graph readout | attention pooling(노드) ⊕ relation histogram ⊕ motif vector | `model.py: BIGNet.forward` |
| 4. Multi-task heads | role/big5/edu·loc·fam/interests 동시 예측 | `model.py` head_* |

### 왜 좋은가 (paper 셀링 포인트)
현재 cosine은 이미 0.99 천장이라 **정확도로는 못 판다.** BIG-Net의 가치는:
1. **해석가능성** — 학습된 `W_r` 크기/영향으로 "어떤 관계가 어떤 persona 차원을 결정하는가"
   를 정량 figure로. + **motif**(관계 2-체인)로 논문의 Association/Characterization 2-mode를 수치화.
2. **추론 효율** — 학습 후엔 LLM 재호출 없이 graph만으로 추론.
3. **Transfer** — 학습 안 한 target LLM 그래프에 zero-shot 적용.

---

## 2. 파일 구성

```
bignet_prototype/
├── persona.py        # 4차원 persona schema + concept 어휘 (정답 구조 정의)
├── datagen.py        # persona → bridging graph 합성 생성기 (forward 시뮬레이션)
├── features.py       # relation histogram + motif vector (readout용 hand-crafted 특징)
├── model.py          # ★ BIG-Net: R-GCN + attention readout + 멀티태스크 헤드 (순수 PyTorch)
├── train.py          # 학습 루프 + 차원별 평가(F1/MAE/Spearman) + ablation
├── sanity_check.py   # (torch 불필요) 데이터에 persona 신호가 실재하는지 sklearn으로 검증
├── selftest_numpy.py # (torch 불필요) BIG-Net forward 수학을 numpy로 1:1 검증
└── requirements.txt
```

> **중요**: `datagen.py`는 합성 데이터다. 실제 연구에서는 이 파일만
> "실제 대화에서 추출한 bridging 관계 로더"로 교체하고, 출력 포맷(`BridgingGraph`
> = node_ids / edge_index / edge_type / labels)만 맞추면 `model.py`·`train.py`는 그대로 재사용된다.

---

## 3. 실행 방법

```bash
pip install -r requirements.txt            # torch는 아래 CPU 설치 권장

# (torch 없이) 데이터 신호 검증 + forward 수학 검증
python sanity_check.py
python selftest_numpy.py

# BIG-Net 학습
python train.py --epochs 30

# Ablation (비평가 방어용)
python train.py --epochs 30 --ablation shuffle_rel      # 관계 유형 무작위화 → 성능 하락 기대
python train.py --epochs 30 --ablation no_handcrafted   # relhist/motif 제거 → 순수 GNN 기여
```

CPU 전용 torch 설치: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 4. 검증 결과 (이 저장소에서 실제 실행)

`sanity_check.py` — 합성 데이터가 복원 가능한 persona 신호를 담는지 확인:
```
(1) FULL(노드+관계):        Role Macro-F1 0.99   ← 노드 정체성이 role을 대부분 설명
(2) STRUCTURE-ONLY(관계만):  Role Macro-F1 0.35   (chance 0.125)
    관계 무작위 재샘플링 시:   Role Macro-F1 0.09   ← 붕괴 = 관계가 담은 고유 신호 실재
```
→ **비평가가 지적한 "관계가 정말 기여하는가"(A8 ablation)** 의 축소판을 데이터가 통과.

`selftest_numpy.py` — BIG-Net forward 수학 검증:
```
R-GCN 벡터화(einsum) vs 브루트포스 최대오차 = 0.00e+00
attention pooling: 그래프별 가중치 합 = 1
헤드 shape/수치 정상 → ALL CHECKS PASSED
```

> 참고: 이 개발 샌드박스는 CPU 전용 torch 휠이 프록시로 차단되어 GNN **학습**은
> 여기서 실행하지 못했다. 대신 (a) 데이터의 학습 가능성(sklearn)과 (b) 모델 forward
> 수학(numpy)을 독립 검증했다. `train.py`는 torch가 설치된 사용자 환경(CPU/GPU)에서 바로 돌아간다.

---

## 5. 다음 단계 (실험계획 노트와 연결)
- `datagen.py` → 실제 PD-Agent 추출 결과 로더로 교체.
- Ablation 확장: relation leave-one-out(A4), centrality 교체(A3), motif 유무(A2).
- 대규모화: `RGCNLayer`를 `torch_geometric.nn.RGCNConv`(+ basis decomposition)로 교체.
- 해석 figure: 학습된 `W_r` norm / attention 분포 → "관계 → persona 차원" 매핑 시각화.
- 관련 노트: `../research_notes/2026-06-10_PersonaBridging_방법론.md`, `..._실험계획.md`
