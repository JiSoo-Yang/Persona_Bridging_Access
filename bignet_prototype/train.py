"""
train.py
========
BIG-Net 학습 + 차원별 평가 + ablation.

실행 예시:
  python train.py --epochs 30                 # 정상 학습
  python train.py --epochs 30 --ablation shuffle_rel      # A8: 관계 유형 무작위화
  python train.py --epochs 30 --ablation no_handcrafted   # relhist/motif 제거(순수 GNN)

평가 메트릭 (논문 E1 재설계 방향과 동일, cosine 대신 차원별):
  Social Role  -> Macro-F1
  Big-Five     -> MAE + 평균 Spearman ρ
  Background   -> edu/loc/fam accuracy
  Interests    -> multi-label micro-F1
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

import persona as P
import datagen
import features as Feat
from model import BIGNet


# ---------------------------------------------------------------------------
# 그래프 리스트 -> 하나의 disjoint 배치 텐서 dict
# ---------------------------------------------------------------------------
def collate(graphs, device, ablation="none", rng=None):
    node_ids, edge_src, edge_dst, edge_type, batch = [], [], [], [], []
    relhist, motif = [], []
    role, big5, edu, loc, fam, interests = [], [], [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        et = g.edge_type.copy()
        if ablation == "shuffle_rel" and rng is not None:
            et = rng.integers(0, P.NUM_RELATIONS, size=et.shape[0])  # 관계 유형 파괴

        N = len(g.node_ids)
        node_ids.append(g.node_ids)
        edge_src.append(g.edge_index[0] + offset)
        edge_dst.append(g.edge_index[1] + offset)
        edge_type.append(et)
        batch.append(np.full(N, gi))
        offset += N

        # hand-crafted 특징 (ablation 시 재계산)
        gg = datagen.BridgingGraph(g.node_ids, g.edge_index, et, g.labels)
        relhist.append(Feat.relation_histogram(et))
        motif.append(Feat.motif_vector(gg.edge_index, et, N))

        role.append(g.labels["role"]); big5.append(g.labels["big5"])
        edu.append(g.labels["education"]); loc.append(g.labels["location"])
        fam.append(g.labels["family"]); interests.append(g.labels["interests"])

    ni = torch.tensor(np.concatenate(node_ids), dtype=torch.long, device=device)
    ei = torch.tensor(np.stack([np.concatenate(edge_src), np.concatenate(edge_dst)]),
                      dtype=torch.long, device=device)
    et_t = torch.tensor(np.concatenate(edge_type), dtype=torch.long, device=device)
    bt = torch.tensor(np.concatenate(batch), dtype=torch.long, device=device)
    in_deg = torch.zeros(offset, device=device).index_add(
        0, ei[1], torch.ones(ei.shape[1], device=device))

    def T(a, dt=torch.float32): return torch.tensor(np.array(a), dtype=dt, device=device)
    return {
        "node_ids": ni, "edge_index": ei, "edge_type": et_t, "in_deg": in_deg,
        "batch": bt, "num_graphs": len(graphs),
        "relhist": T(relhist), "motif": T(motif),
        "role": T(role, torch.long), "big5": T(big5),
        "edu": T(edu, torch.long), "loc": T(loc, torch.long), "fam": T(fam, torch.long),
        "interests": T(interests),
    }


def compute_loss(out, b):
    l = Fnn.cross_entropy(out["role"], b["role"])
    l = l + Fnn.mse_loss(out["big5"], b["big5"])
    l = l + Fnn.cross_entropy(out["edu"], b["edu"])
    l = l + Fnn.cross_entropy(out["loc"], b["loc"])
    l = l + Fnn.cross_entropy(out["fam"], b["fam"])
    l = l + Fnn.binary_cross_entropy_with_logits(out["interests"], b["interests"])
    return l


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    out = model(batch)
    role_pred = out["role"].argmax(1).cpu().numpy()
    role_true = batch["role"].cpu().numpy()
    role_f1 = f1_score(role_true, role_pred, average="macro")

    big5_pred = out["big5"].cpu().numpy(); big5_true = batch["big5"].cpu().numpy()
    big5_mae = np.abs(big5_pred - big5_true).mean()
    rhos = [spearmanr(big5_pred[:, k], big5_true[:, k]).correlation for k in range(P.NUM_BIG5)]
    big5_rho = float(np.nanmean(rhos))

    def acc(key):
        return (out[key].argmax(1).cpu().numpy() == batch[key].cpu().numpy()).mean()
    bg_acc = np.mean([acc("edu"), acc("loc"), acc("fam")])

    int_pred = (torch.sigmoid(out["interests"]).cpu().numpy() > 0.5).astype(int)
    int_f1 = f1_score(batch["interests"].cpu().numpy(), int_pred, average="micro", zero_division=0)
    return dict(role_f1=role_f1, big5_mae=big5_mae, big5_rho=big5_rho,
                bg_acc=bg_acc, int_f1=int_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ablation", choices=["none", "shuffle_rel", "no_handcrafted"], default="none")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    print(f"device={device}  ablation={args.ablation}")

    train_g = datagen.make_dataset(args.n_train, seed=1)
    test_g = datagen.make_dataset(args.n_test, seed=2)
    train_b = collate(train_g, device, args.ablation, rng)
    test_b = collate(test_g, device, args.ablation, rng)

    model = BIGNet(hidden=args.hidden, num_layers=args.layers).to(device)
    if args.ablation == "no_handcrafted":   # relhist/motif 기여 제거
        train_b["relhist"].zero_(); train_b["motif"].zero_()
        test_b["relhist"].zero_(); test_b["motif"].zero_()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = compute_loss(model(train_b), train_b)
        loss.backward(); opt.step()
        if ep % max(1, args.epochs // 6) == 0 or ep == 1:
            m = evaluate(model, test_b)
            print(f"ep{ep:3d} loss={loss.item():.3f} | "
                  f"Role-F1={m['role_f1']:.3f} Big5-MAE={m['big5_mae']:.3f} "
                  f"Big5-rho={m['big5_rho']:.3f} BG-acc={m['bg_acc']:.3f} Int-F1={m['int_f1']:.3f}")

    m = evaluate(model, test_b)
    print("\n=== FINAL (test) ===")
    print(f"Social Role  Macro-F1 : {m['role_f1']:.3f}   (chance {1/P.NUM_ROLES:.3f})")
    print(f"Big-Five     MAE      : {m['big5_mae']:.3f}   Spearman rho: {m['big5_rho']:.3f}")
    print(f"Background    accuracy : {m['bg_acc']:.3f}")
    print(f"Interests    micro-F1 : {m['int_f1']:.3f}")


if __name__ == "__main__":
    main()
