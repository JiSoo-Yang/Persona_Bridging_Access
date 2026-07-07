"""
datagen.py
==========
합성 Bridging Graph 생성기:  persona  --(생성 과정)-->  directed typed multigraph

이것은 PD-Agent 파이프라인의 '순방향(forward)'을 시뮬레이션합니다:
    persona 부여 -> (대화) -> bridging 관계 추출 -> semantic graph
BIG-Net은 이 graph만 보고 persona를 '역방향(inverse)'으로 복원하도록 학습됩니다.

왜 합성인가?
------------
방법론 노트대로 persona를 직접 부여하므로 (graph, persona) 쌍을 무제한 만들 수 있습니다.
실제 연구에서는 이 파일을 "실제 대화에서 추출한 bridging 관계 로더"로 교체하면 됩니다.
(출력 포맷 = BridgingGraph dataclass 만 맞추면 model/train은 그대로 재사용됩니다.)

각 그래프는 다음을 담습니다:
  - node_ids   : 각 노드의 concept vocab index         (shape [N])
  - edge_index : 방향성 엣지 (src, dst)                  (shape [2, E])
  - edge_type  : 각 엣지의 관계 유형 0..6                (shape [E])
  - labels     : 4차원 persona 정답 (dict)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import persona as P


@dataclass
class BridgingGraph:
    node_ids: np.ndarray     # [N] int   (concept vocab index)
    edge_index: np.ndarray   # [2, E] int
    edge_type: np.ndarray    # [E] int   (0..6)
    labels: dict             # persona ground-truth


# ---------------------------------------------------------------------------
# 1) persona 샘플링
# ---------------------------------------------------------------------------
def sample_persona(rng: np.random.Generator) -> dict:
    role_idx = rng.integers(P.NUM_ROLES)
    role = P.SOCIAL_ROLES[role_idx]
    struct = P.ROLE_STRUCTURAL[role]

    # Big-Five: role 성향과 약하게 상관 (구조적 role -> 높은 성실성/낮은 개방성 경향)
    big5 = rng.uniform(0.2, 0.8, size=P.NUM_BIG5)
    big5[1] = np.clip(0.5 + 0.3 * (struct - 0.5) + rng.normal(0, 0.1), 0, 1)   # Conscientiousness↑ (구조적)
    big5[0] = np.clip(0.5 - 0.3 * (struct - 0.5) + rng.normal(0, 0.1), 0, 1)   # Openness↑ (서사적)

    edu = rng.integers(len(P.EDUCATION))
    loc = rng.integers(len(P.LOCATION))
    fam = rng.integers(len(P.FAMILY))

    # Interests multi-label: 각 라벨을 베르누이로. hobby 1~3개, value 1~2개 정도.
    interests = np.zeros(P.NUM_INTERESTS, dtype=np.float32)
    n_hobby = rng.integers(1, 4)
    for j in rng.choice(len(P.HOBBIES), size=n_hobby, replace=False):
        interests[j] = 1.0
    n_val = rng.integers(1, 3)
    for j in rng.choice(len(P.VALUES), size=n_val, replace=False):
        interests[len(P.HOBBIES) + j] = 1.0
    # comm style: 구조적 role -> direct 경향
    comm = 0 if rng.uniform() < (0.3 + 0.5 * struct) else 1
    interests[len(P.HOBBIES) + len(P.VALUES) + comm] = 1.0

    return {
        "role": int(role_idx), "role_name": role, "structural": float(struct),
        "big5": big5.astype(np.float32),
        "education": int(edu), "location": int(loc), "family": int(fam),
        "interests": interests,
    }


# ---------------------------------------------------------------------------
# 2) persona -> 관계 유형 분포 p(7-dim)   (논문 §3.2 가설을 인코딩)
# ---------------------------------------------------------------------------
def relation_distribution(persona: dict, rng: np.random.Generator) -> np.ndarray:
    struct = persona["structural"]
    openness = persona["big5"][0]
    consci = persona["big5"][1]

    logits = np.zeros(P.NUM_RELATIONS)
    # Association(part-of, member-of): 구조성/성실성이 높을수록 ↑
    assoc_boost = 1.6 * (struct - 0.5) + 0.8 * (consci - 0.5)
    for r in P.ASSOCIATION_RELS:
        logits[r] += assoc_boost
    # Characterization(instrument..temporal): 서사성/개방성이 높을수록 ↑
    char_boost = 1.6 * (0.5 - struct) + 0.8 * (openness - 0.5)
    for r in P.CHARACTERIZATION_RELS:
        logits[r] += char_boost
    logits += rng.normal(0, 0.3, size=P.NUM_RELATIONS)     # 노이즈
    p = np.exp(logits - logits.max())
    return p / p.sum()


# ---------------------------------------------------------------------------
# 3) persona -> concept 노드 집합
# ---------------------------------------------------------------------------
def sample_concepts(persona: dict, rng: np.random.Generator, n_nodes: int) -> list[int]:
    pool = []
    pool += P.ROLE_CONCEPTS[persona["role_name"]] * 3          # role 개념 강한 신호
    active = np.where(persona["interests"] == 1.0)[0]
    for j in active:
        label = P.INTEREST_LABELS[j]
        if label in P.INTEREST_CONCEPTS:
            pool += P.INTEREST_CONCEPTS[label] * 2             # interest 개념 중간 신호
    # 교육수준이 높을수록 추상 generic 개념 조금 더 (약한 background 신호)
    n_generic = 4 + persona["education"]
    pool += list(rng.choice(P.GENERIC_CONCEPTS, size=n_generic))

    concepts = list(rng.choice(pool, size=min(n_nodes, len(pool)), replace=False)) \
        if len(pool) >= n_nodes else pool + list(rng.choice(P.GENERIC_CONCEPTS, size=n_nodes - len(pool)))
    idxs = [P.CONCEPT2IDX[c] for c in concepts[:n_nodes]]
    return idxs


# ---------------------------------------------------------------------------
# 4) 전체 그래프 생성
# ---------------------------------------------------------------------------
def make_graph(rng: np.random.Generator,
               n_nodes_range=(8, 16),
               edges_per_node=(1.5, 2.5)) -> BridgingGraph:
    persona = sample_persona(rng)
    n_nodes = int(rng.integers(*n_nodes_range))
    node_ids = np.array(sample_concepts(persona, rng, n_nodes), dtype=np.int64)
    N = len(node_ids)

    p_rel = relation_distribution(persona, rng)
    n_edges = max(N, int(N * rng.uniform(*edges_per_node)))

    src, dst, etype = [], [], []
    for _ in range(n_edges):
        r = int(rng.choice(P.NUM_RELATIONS, p=p_rel))
        a, b = rng.integers(N), rng.integers(N)
        if a == b:
            b = (b + 1) % N
        # Association 관계는 방향성(part->whole)이 있으니 그대로, 나머지도 방향 유지
        src.append(a); dst.append(b); etype.append(r)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_type = np.array(etype, dtype=np.int64)

    labels = {
        "role": persona["role"], "big5": persona["big5"],
        "education": persona["education"], "location": persona["location"],
        "family": persona["family"], "interests": persona["interests"],
    }
    return BridgingGraph(node_ids, edge_index, edge_type, labels)


def make_dataset(n: int, seed: int = 0) -> list[BridgingGraph]:
    rng = np.random.default_rng(seed)
    return [make_graph(rng) for _ in range(n)]


if __name__ == "__main__":
    ds = make_dataset(5, seed=1)
    for g in ds:
        print(f"role={P.SOCIAL_ROLES[g.labels['role']]:18s} "
              f"N={len(g.node_ids):2d} E={g.edge_type.shape[0]:2d} "
              f"rel_hist={np.bincount(g.edge_type, minlength=P.NUM_RELATIONS)}")
