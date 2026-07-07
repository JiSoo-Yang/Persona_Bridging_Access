"""
features.py
===========
Graph-level '손으로 만든(hand-crafted)' 특징. BIG-Net의 readout에서 GNN 임베딩과 concat됩니다.

두 가지:
  1) relation histogram : 7개 관계 유형의 정규화 빈도  -> [7]
     (Association vs Characterization 비율이 persona mode의 강한 신호)
  2) motif vector       : 3-노드 관계 체인(triad) 패턴의 빈도 -> [K]
     예: part-of -> member-of (구조적 분해),  cause-of -> temporal (서사적 추론)
     논문의 Association/Characterization 2-mode를 정량화하는 역할.

이 특징들은 numpy만으로 계산되며, 학습 없이도 persona 신호를 상당 부분 담고 있어
'데이터가 실제로 복원 가능한 신호를 갖는지' 검증(sanity check)에도 쓰입니다.
"""
from __future__ import annotations
import numpy as np
import persona as P


def relation_histogram(edge_type: np.ndarray) -> np.ndarray:
    h = np.bincount(edge_type, minlength=P.NUM_RELATIONS).astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


# 관심 있는 2-엣지 motif: (관계r1 로 들어온 노드가 관계r2 로 나감)
# 전부 49개(7x7) 대신, 이론적으로 의미있는 소수 + 집계값으로 압축.
def motif_vector(edge_index: np.ndarray, edge_type: np.ndarray, n_nodes: int) -> np.ndarray:
    """길이-2 방향 경로 a --r1--> b --r2--> c 의 (r1,r2) 유형쌍을 센다."""
    # 노드별 나가는 엣지 목록
    out_edges = [[] for _ in range(n_nodes)]
    E = edge_type.shape[0]
    for e in range(E):
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        out_edges[s].append((d, int(edge_type[e])))

    pair_counts = np.zeros((P.NUM_RELATIONS, P.NUM_RELATIONS), dtype=np.float32)
    for e in range(E):
        b, r1 = int(edge_index[1, e]), int(edge_type[e])
        for (_c, r2) in out_edges[b]:
            pair_counts[r1, r2] += 1.0

    total = pair_counts.sum()
    if total > 0:
        pair_counts /= total

    # 압축 특징: 4개 모드-쌍 집계 + 대각(자기 반복) 합
    A, C = P.ASSOCIATION_RELS, P.CHARACTERIZATION_RELS
    aa = pair_counts[np.ix_(A, A)].sum()   # 구조->구조 (분해 체인)
    ac = pair_counts[np.ix_(A, C)].sum()
    ca = pair_counts[np.ix_(C, A)].sum()
    cc = pair_counts[np.ix_(C, C)].sum()   # 서사->서사 (추론 체인)
    diag = float(np.trace(pair_counts))
    return np.array([aa, ac, ca, cc, diag], dtype=np.float32)


MOTIF_DIM = 5


def graph_feature_vector(g) -> np.ndarray:
    """relation hist(7) + motif(5) + node-bag(vocab) 를 이어붙인 sanity-check용 벡터."""
    rh = relation_histogram(g.edge_type)
    mv = motif_vector(g.edge_index, g.edge_type, len(g.node_ids))
    bag = np.bincount(g.node_ids, minlength=P.VOCAB_SIZE).astype(np.float32)
    bag = bag / bag.sum() if bag.sum() > 0 else bag
    return np.concatenate([rh, mv, bag])


if __name__ == "__main__":
    import datagen
    g = datagen.make_dataset(1, seed=3)[0]
    print("relation hist:", relation_histogram(g.edge_type).round(2))
    print("motif vector :", motif_vector(g.edge_index, g.edge_type, len(g.node_ids)).round(2))
    print("full feat dim:", graph_feature_vector(g).shape)
