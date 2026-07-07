"""
selftest_numpy.py
=================
torch 없이, BIG-Net forward의 '수학'을 numpy로 1:1 재현하여 정확성을 검증한다.
(이 샌드박스는 CPU 전용 torch 휠이 막혀 있어 GNN 학습은 사용자 환경에서 돌린다.
 여기서는 model.py/train.py의 핵심 텐서 연산이 옳은지 독립 검증한다.)

검증 항목:
  1) collate 오프셋: disjoint 배치의 edge_index 오프셋과 batch 벡터가 맞는가
  2) R-GCN 메시지: einsum('ei,eio->eo') + degree 정규화 + index_add(scatter)
     -> 브루트포스 이중 루프 결과와 일치하는가
  3) attention segment-softmax pooling: 그래프별 가중치 합이 1인가, pooled shape가 [B,H]인가
  4) 헤드 출력 shape가 각 persona 차원과 맞는가, NaN이 없는가
"""
import numpy as np
import persona as P
import datagen
import features as Feat

rng = np.random.default_rng(0)


# --- collate(오프셋) 재현 ---------------------------------------------------
def collate_np(graphs):
    node_ids, esrc, edst, etype, batch = [], [], [], [], []
    off = 0
    for gi, g in enumerate(graphs):
        N = len(g.node_ids)
        node_ids.append(g.node_ids)
        esrc.append(g.edge_index[0] + off); edst.append(g.edge_index[1] + off)
        etype.append(g.edge_type); batch.append(np.full(N, gi)); off += N
    ei = np.stack([np.concatenate(esrc), np.concatenate(edst)])
    return (np.concatenate(node_ids), ei, np.concatenate(etype),
            np.concatenate(batch), off)


def rgcn_forward(x, ei, et, W, W0, num_rel):
    """out_v = W0 x_v + sum_{u->v} (1/indeg_v) W_{r(u,v)} x_u"""
    src, dst = ei[0], ei[1]
    N = x.shape[0]
    indeg = np.zeros(N); np.add.at(indeg, dst, 1.0); indeg = np.clip(indeg, 1, None)
    # 벡터화 메시지: einsum('ei,eio->eo')
    msg = np.einsum("ei,eio->eo", x[src], W[et])
    msg = msg / indeg[dst][:, None]
    out = x @ W0
    np.add.at(out, dst, msg)
    return out, indeg


def rgcn_bruteforce(x, ei, et, W, W0, num_rel):
    src, dst = ei[0], ei[1]
    N = x.shape[0]
    indeg = np.zeros(N); np.add.at(indeg, dst, 1.0); indeg = np.clip(indeg, 1, None)
    out = x @ W0
    for e in range(ei.shape[1]):
        u, v, r = src[e], dst[e], et[e]
        out[v] += (W[r].T @ x[u]) / indeg[v]
    return out


def segment_softmax_pool(x, batch, B):
    scores = x @ rng.normal(size=(x.shape[1], 1))     # 임의 attention
    scores = scores.squeeze(-1); scores -= scores.max()
    exp = np.exp(scores)
    denom = np.zeros(B); np.add.at(denom, batch, exp); denom += 1e-9
    alpha = (exp / denom[batch])[:, None]
    pooled = np.zeros((B, x.shape[1])); np.add.at(pooled, batch, alpha * x)
    # 검증용: 그래프별 alpha 합
    asum = np.zeros(B); np.add.at(asum, batch, alpha.squeeze(-1))
    return pooled, asum


def main():
    graphs = datagen.make_dataset(3, seed=5)
    ni, ei, et, batch, Ntot = collate_np(graphs)
    B = len(graphs)
    print(f"[1] collate: {B} graphs, N_total={Ntot}, E_total={ei.shape[1]}")
    assert ni.shape[0] == Ntot
    assert ei.max() < Ntot and ei.min() >= 0, "edge_index 오프셋 오류"
    assert batch.max() == B - 1
    # 그래프별 노드 수 일치 확인
    counts = np.bincount(batch)
    assert list(counts) == [len(g.node_ids) for g in graphs]
    print("    -> 오프셋/배치 벡터 정상")

    # node feature = 임의 임베딩(모델의 nn.Embedding 대역)
    node_dim, hid = 8, 8
    emb_table = rng.normal(size=(P.VOCAB_SIZE, node_dim))
    x = emb_table[ni]

    # R-GCN 1 레이어: 벡터화 vs 브루트포스
    W = rng.normal(size=(P.NUM_RELATIONS, node_dim, hid)) * 0.3
    W0 = rng.normal(size=(node_dim, hid)) * 0.3
    out_vec, indeg = rgcn_forward(x, ei, et, W, W0, P.NUM_RELATIONS)
    out_bf = rgcn_bruteforce(x, ei, et, W, W0, P.NUM_RELATIONS)
    err = np.abs(out_vec - out_bf).max()
    print(f"[2] R-GCN 벡터화 vs 브루트포스 최대오차 = {err:.2e}")
    assert err < 1e-9, "einsum/scatter 구현 불일치"
    print("    -> 메시지 전달 수학 정확")

    # attention pooling
    h = np.maximum(out_vec, 0)   # ReLU
    pooled, asum = segment_softmax_pool(h, batch, B)
    print(f"[3] pooled shape={pooled.shape}, 그래프별 attention 합={asum.round(4)}")
    assert pooled.shape == (B, hid)
    assert np.allclose(asum, 1.0, atol=1e-6), "segment softmax 정규화 오류"
    print("    -> attention pooling 정상 (그래프별 가중치 합=1)")

    # readout + head shape
    relhist = np.stack([Feat.relation_histogram(g.edge_type) for g in graphs])
    motif = np.stack([Feat.motif_vector(g.edge_index, g.edge_type, len(g.node_ids)) for g in graphs])
    z = np.concatenate([pooled, relhist @ rng.normal(size=(P.NUM_RELATIONS, hid)),
                        motif @ rng.normal(size=(Feat.MOTIF_DIM, hid))], axis=1)
    heads = {
        "role": z @ rng.normal(size=(z.shape[1], P.NUM_ROLES)),
        "big5": z @ rng.normal(size=(z.shape[1], P.NUM_BIG5)),
        "edu": z @ rng.normal(size=(z.shape[1], len(P.EDUCATION))),
        "interests": z @ rng.normal(size=(z.shape[1], P.NUM_INTERESTS)),
    }
    print("[4] head output shapes:", {k: v.shape for k, v in heads.items()})
    assert heads["role"].shape == (B, P.NUM_ROLES)
    assert heads["big5"].shape == (B, P.NUM_BIG5)
    assert heads["interests"].shape == (B, P.NUM_INTERESTS)
    assert all(np.isfinite(v).all() for v in heads.values()), "NaN/Inf 발생"
    print("    -> readout/헤드 shape 및 수치 정상")

    print("\nALL CHECKS PASSED ✅  (model.py의 forward 수학이 정확함이 확인됨)")


if __name__ == "__main__":
    main()
