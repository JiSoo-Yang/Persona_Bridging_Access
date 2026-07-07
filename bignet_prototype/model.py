"""
model.py
========
BIG-Net (Bridging-Inference Graph Network) — 순수 PyTorch 구현 (torch_geometric 불필요).

구성 (방법론 노트 §2.2와 동일):
  [Stage 1] Node embedding   : concept id -> 벡터 (nn.Embedding; 실제로는 문장 임베더로 교체 가능)
  [Stage 2] Relational GNN   : 관계 유형(7종)마다 별도 가중치 W_r 를 쓰는 R-GCN 레이어 L개
  [Stage 3] Graph readout     : attention pooling(노드 임베딩) ⊕ relation histogram ⊕ motif vector
  [Stage 4] Multi-task heads  : role(CE) / big5(MSE) / edu·loc·fam(CE) / interests(BCE)

--------------------------------------------------------------------------
왜 R-GCN인가? (BIG-Net을 처음 보는 사람을 위한 핵심)
--------------------------------------------------------------------------
일반 GCN은 이웃 노드의 정보를 '똑같은' 가중치로 섞습니다. 하지만 bridging graph의
엣지는 7가지 '유형'(part-of, cause-of, ...)이 있고, 이 유형 분포가 곧 persona 신호입니다.
R-GCN(Relational GCN, Schlichtkrull et al. 2018)은 유형 r마다 별도 변환 W_r 를 두어,
"어떤 관계로 연결됐는가"에 따라 메시지를 다르게 전달합니다. 학습 후 W_r 의 크기/영향력을
보면 "어떤 관계가 어떤 persona 차원을 결정하는가"를 해석할 수 있습니다.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

import persona as P
from features import MOTIF_DIM


# ---------------------------------------------------------------------------
# Relational GCN layer
# ---------------------------------------------------------------------------
class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_rel: int):
        super().__init__()
        self.num_rel = num_rel
        # 관계별 가중치 W_r : [num_rel, in_dim, out_dim]
        self.W = nn.Parameter(torch.empty(num_rel, in_dim, out_dim))
        self.W0 = nn.Linear(in_dim, out_dim)          # self-loop 변환
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, edge_index, edge_type, in_deg):
        # x: [N, in]  edge_index: [2, E]  edge_type: [E]  in_deg: [N]
        out = self.W0(x)                               # 자기 자신
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            # 메시지: x[src] 를 각 엣지의 관계 가중치로 변환  -> [E, out]
            msg = torch.einsum("ei,eio->eo", x[src], self.W[edge_type])
            # 목적지 노드 in-degree 로 정규화
            norm = 1.0 / in_deg[dst].clamp(min=1).unsqueeze(1)
            msg = msg * norm
            out = out.index_add(0, dst, msg)
        return out + self.bias


# ---------------------------------------------------------------------------
# BIG-Net
# ---------------------------------------------------------------------------
class BIGNet(nn.Module):
    def __init__(self, vocab_size=P.VOCAB_SIZE, node_dim=64, hidden=64,
                 num_rel=P.NUM_RELATIONS, num_layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, node_dim)
        dims = [node_dim] + [hidden] * num_layers
        self.layers = nn.ModuleList(
            [RGCNLayer(dims[i], dims[i + 1], num_rel) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # attention pooling
        self.att = nn.Linear(hidden, 1)
        # hand-crafted 특징 투영
        self.relhist_proj = nn.Linear(num_rel, hidden)
        self.motif_proj = nn.Linear(MOTIF_DIM, hidden)

        readout = hidden * 3
        # multi-task heads
        self.head_role = nn.Linear(readout, P.NUM_ROLES)
        self.head_big5 = nn.Linear(readout, P.NUM_BIG5)
        self.head_edu = nn.Linear(readout, len(P.EDUCATION))
        self.head_loc = nn.Linear(readout, len(P.LOCATION))
        self.head_fam = nn.Linear(readout, len(P.FAMILY))
        self.head_int = nn.Linear(readout, P.NUM_INTERESTS)

    def forward(self, batch):
        """batch: collate()가 만든 dict. 하나의 disjoint 큰 그래프로 처리."""
        x = self.emb(batch["node_ids"])                 # [N, node_dim]
        ei, et, indeg = batch["edge_index"], batch["edge_type"], batch["in_deg"]
        for layer in self.layers:
            x = Fnn.relu(layer(x, ei, et, indeg))
            x = self.dropout(x)

        # --- 그래프별 attention pooling (segment softmax) ---
        bt = batch["batch"]                             # [N] 노드->그래프 index
        B = batch["num_graphs"]
        scores = self.att(x).squeeze(-1)                # [N]
        scores = scores - scores.max()
        exp = scores.exp()
        denom = torch.zeros(B, device=x.device).index_add(0, bt, exp) + 1e-9
        alpha = (exp / denom[bt]).unsqueeze(-1)         # [N,1]
        pooled = torch.zeros(B, x.shape[1], device=x.device).index_add(0, bt, alpha * x)

        # --- hand-crafted 특징 결합 ---
        z = torch.cat([pooled,
                       Fnn.relu(self.relhist_proj(batch["relhist"])),
                       Fnn.relu(self.motif_proj(batch["motif"]))], dim=-1)

        return {
            "role": self.head_role(z), "big5": torch.sigmoid(self.head_big5(z)),
            "edu": self.head_edu(z), "loc": self.head_loc(z), "fam": self.head_fam(z),
            "interests": self.head_int(z),
        }


if __name__ == "__main__":
    m = BIGNet()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"BIG-Net instantiated. trainable params = {n_params:,}")
