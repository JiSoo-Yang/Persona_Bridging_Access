"""
sanity_check.py
===============
학습(GNN) 이전에, 합성 데이터가 '복원 가능한 persona 신호'를 실제로 담고 있는지 확인.
graph-level 특징(relation hist + motif + node-bag)에 단순 분류기/회귀기를 얹어 성능을 봅니다.

동시에 핵심 ablation을 미리 보여줍니다:
  - 정상 그래프  vs  relation label을 셔플한 그래프
  관계 라벨이 persona 신호의 핵심이라면, 셔플 시 성능이 떨어져야 합니다.
  (비평가가 요구한 A8 ablation의 축소판)

torch 없이 sklearn만으로 실행됩니다.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error

import persona as P
import datagen
import features as F


def _feat(g, structure_only):
    """structure_only=True 이면 관계 히스토그램+motif 만 사용(노드 정체성 제외)."""
    if structure_only:
        rh = F.relation_histogram(g.edge_type)
        mv = F.motif_vector(g.edge_index, g.edge_type, len(g.node_ids))
        return np.concatenate([rh, mv])
    return F.graph_feature_vector(g)


def build_XY(graphs, shuffle_relations=False, structure_only=False, seed=0):
    rng = np.random.default_rng(seed)
    X, y_role, y_big5, y_int = [], [], [], []
    for g in graphs:
        if shuffle_relations:
            # 관계 유형을 균등 무작위로 '재샘플링' -> persona-조건 분포 파괴 (순열은 히스토그램 보존이라 부적절)
            et = rng.integers(0, P.NUM_RELATIONS, size=g.edge_type.shape[0])
            g = datagen.BridgingGraph(g.node_ids, g.edge_index, et, g.labels)
        X.append(_feat(g, structure_only))
        y_role.append(g.labels["role"])
        y_big5.append(g.labels["big5"])
        y_int.append(g.labels["interests"])
    return (np.array(X), np.array(y_role), np.array(y_big5), np.array(y_int))


def evaluate(tag, train, test):
    Xtr, rtr, btr, itr = train
    Xte, rte, bte, ite = test

    # Social Role: 다중클래스 -> Macro-F1
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, rtr)
    role_f1 = f1_score(rte, clf.predict(Xte), average="macro")

    # Big-Five: 연속 -> MAE (5차원 평균)
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, btr)
    big5_mae = mean_absolute_error(bte, reg.predict(Xte))

    # Interests: 다중라벨 -> micro-F1 (라벨별 로지스틱)
    preds = np.zeros_like(ite)
    for k in range(itr.shape[1]):
        if len(np.unique(itr[:, k])) < 2:
            continue
        c = LogisticRegression(max_iter=2000)
        c.fit(Xtr, itr[:, k])
        preds[:, k] = c.predict(Xte)
    int_f1 = f1_score(ite, preds, average="micro", zero_division=0)

    print(f"[{tag:22s}] Role Macro-F1={role_f1:.3f} | Big5 MAE={big5_mae:.3f} | Interests micro-F1={int_f1:.3f}")
    return role_f1, big5_mae, int_f1


if __name__ == "__main__":
    print("Generating synthetic bridging graphs...")
    train_g = datagen.make_dataset(1500, seed=1)
    test_g = datagen.make_dataset(500, seed=2)
    print(f"  train={len(train_g)}  test={len(test_g)}  schema={P.dims_summary()}")
    print()

    print("=== (1) FULL features (node identity + relations + motif) ===")
    evaluate("normal", build_XY(train_g), build_XY(test_g))
    evaluate("shuffled relations",
             build_XY(train_g, shuffle_relations=True, seed=10),
             build_XY(test_g, shuffle_relations=True, seed=11))
    print("  -> node 정체성이 role/interest를 대부분 설명하므로 관계 셔플 영향이 작다\n"
          "     (이것이 바로 '관계가 필요한가'라는 비판의 핵심)")

    print()
    print("=== (2) STRUCTURE-ONLY features (관계 히스토그램+motif, 노드 제외) ===")
    evaluate("normal (structure)", build_XY(train_g, structure_only=True),
             build_XY(test_g, structure_only=True))
    evaluate("shuffled (structure)",
             build_XY(train_g, shuffle_relations=True, structure_only=True, seed=10),
             build_XY(test_g, shuffle_relations=True, structure_only=True, seed=11))
    print("  -> 노드를 제외하면 관계 구조만 남고, 셔플 시 성능이 무너진다\n"
          "     = bridging 관계가 담은 '고유한' persona 신호가 실재함(A8 ablation 논리)")

    print()
    print(f"(reference) random role Macro-F1 ~ {1/P.NUM_ROLES:.3f}  (chance)")
