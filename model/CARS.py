
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Basic convolutional block used by the lightweight shared backbone."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Lightweight CNN encoder. Keeps spatial resolution and returns 4 feature levels."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 64):
        super().__init__()
        c1 = feat_dim // 2
        c2 = feat_dim // 2
        c3 = feat_dim
        c4 = feat_dim
        self.conv1 = ConvBNReLU(in_channels, c1)
        self.conv2 = ConvBNReLU(c1, c2)
        self.conv3 = ConvBNReLU(c2, c3)
        self.conv4 = ConvBNReLU(c3, c4)
        self.out_channels_list = [c1, c2, c3, c4]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # All four stages preserve the patch resolution. This allows every level to
        # describe local structure over the same spatial support before compact pooling.
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        if return_features:
            return f4, [f1, f2, f3, f4]
        return f4


class CompactRelationLayer(nn.Module):
    """
    Converts a feature map into K compact structural tokens and a KxK relation matrix.

    Feature map F_l: B x C x H x W
      -> projected pixel embeddings X_l: B x N x D
      -> query-attentive compact tokens Z_l: B x K x D
      -> structural relation R_l = softmax(Z_l Z_l^T / tau): B x K x K

    This replaces discrete token assignment / codebook CE with relation-level encoding.
    """

    def __init__(
        self,
        in_channels: int,
        rel_dim: int = 128,
        num_tokens: int = 16,
        attn_temp: float = 0.2,
        relation_temp: float = 0.2,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.rel_dim = int(rel_dim)
        self.attn_temp = float(attn_temp)
        self.relation_temp = float(relation_temp)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, rel_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(rel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(rel_dim, rel_dim, kernel_size=1, bias=True),
        )
        self.queries = nn.Parameter(torch.randn(num_tokens, rel_dim) * 0.02)
        self.token_norm = nn.LayerNorm(rel_dim)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = feat.shape

        # Equation (1): project the level-specific CNN feature into a common D-dimensional
        # embedding space, flatten HxW positions into N spatial tokens, and normalize them.
        x = self.proj(feat)  # B,D,H,W
        x = x.flatten(2).transpose(1, 2).contiguous()  # B,N,D
        x = F.normalize(x, dim=-1)
        q = F.normalize(self.queries, dim=-1)  # K,D

        # Query-specific spatial aggregation. Each structural token attends to all patch pixels.
        # logits: B,K,N; attn normalized over spatial positions.
        logits = torch.einsum("bnd,kd->bkn", x, q) / max(self.attn_temp, 1e-6)
        attn = F.softmax(logits, dim=-1)
        tokens = torch.einsum("bkn,bnd->bkd", attn, x)
        tokens = self.token_norm(tokens)
        tokens = F.normalize(tokens, dim=-1)

        # Equation (3): pairwise compact-token affinities form a KxK structural graph.
        # Row-wise softmax makes every row a relation distribution suitable for KL loss.
        sim = torch.bmm(tokens, tokens.transpose(1, 2)) / max(self.relation_temp, 1e-6)
        relation = F.softmax(sim, dim=-1)
        return {"tokens": tokens, "relation": relation, "attn": attn.view(B, self.num_tokens, H, W)}


class MultiLevelCompactRelationEncoder(nn.Module):
    """Applies CompactRelationLayer to each backbone stage."""

    def __init__(
        self,
        channels_list: List[int],
        rel_dim: int = 128,
        num_tokens: int = 16,
        attn_temp: float = 0.2,
        relation_temp: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CompactRelationLayer(c, rel_dim, num_tokens, attn_temp, relation_temp)
            for c in channels_list
        ])

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        # Apply the same relation-construction principle independently at each backbone level.
        tokens, relations, attns = [], [], []
        for f, layer in zip(feats, self.layers):
            out = layer(f)
            tokens.append(out["tokens"])
            relations.append(out["relation"])
            attns.append(out["attn"])
        return {"tokens": tokens, "relations": relations, "attns": attns}


class CARSStudentTeacher(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 64,
        rel_dim: int = 128,
        num_relation_tokens: int = 16,
        attn_temp: float = 0.2,
        relation_temp: float = 0.2,
        teacher_momentum: float = 0.996,
    ):
        super().__init__()
        self.student_backbone = Encoder(in_channels=in_channels, feat_dim=feat_dim)
        self.student_relation = MultiLevelCompactRelationEncoder(
            self.student_backbone.out_channels_list,
            rel_dim=rel_dim,
            num_tokens=num_relation_tokens,
            attn_temp=attn_temp,
            relation_temp=relation_temp,
        )
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_relation = copy.deepcopy(self.student_relation)
        self.teacher_momentum = float(teacher_momentum)
        self._freeze_teacher()

    def _freeze_teacher(self):
        # The teacher is a target generator only. Gradients update the student, while the
        # teacher parameters evolve exclusively through the EMA update in Equation (5).
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_relation.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, m: Optional[float] = None):
        """Apply the EMA teacher update from Equation (5)."""
        if m is None:
            m = self.teacher_momentum
        for s, t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)
        for s, t in zip(self.student_relation.parameters(), self.teacher_relation.parameters()):
            t.data.mul_(m).add_(s.data, alpha=1.0 - m)

    def student_forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Return dense features, pooled features, compact tokens, and relations."""
        feat, feats = self.student_backbone(x, return_features=True)
        rel = self.student_relation(feats)
        pooled = F.normalize(feat.mean(dim=[2, 3]), dim=-1)
        return {
            "feat": feat,
            "feats": feats,
            "pooled": pooled,
            "tokens": rel["tokens"],
            "relations": rel["relations"],
            "attns": rel["attns"],
        }

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor) -> Dict[str, object]:
        """Generate stop-gradient targets and teacher-side reliability cues."""
        feat, feats = self.teacher_backbone(x, return_features=True)
        rel = self.teacher_relation(feats)
        pooled = F.normalize(feat.mean(dim=[2, 3]), dim=-1)
        return {
            "feat": feat,
            "feats": feats,
            "pooled": pooled,
            "tokens": rel["tokens"],
            "relations": rel["relations"],
            "attns": rel["attns"],
        }

    @torch.no_grad()
    def inference_pair(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the feature and relation discrepancy cues used by ARCS."""
        o1 = self.student_forward(x1)
        o2 = self.student_forward(x2)
        d_feat = feature_discrepancy(o1["pooled"], o2["pooled"])
        d_rel = relation_discrepancy(o1["relations"], o2["relations"])
        return {"d_feat": d_feat, "d_rel": d_rel, "pooled1": o1["pooled"], "pooled2": o2["pooled"]}


# -------------------------
# Loss / metric utilities
# -------------------------
def safe_mean(values: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mask-aware mean that returns a differentiable zero when no anchor is selected."""
    values = values.float()
    if weight is None:
        return values.mean()
    w = weight.detach().float().view_as(values)
    if float(w.sum().detach().cpu()) <= 1e-8:
        return values.mean() * 0.0
    return (values * w).sum() / (w.sum() + 1e-8)


def feature_discrepancy(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Squared distance between normalized pooled features (Equation (6))."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return ((z1 - z2) ** 2).sum(dim=-1)


def _relation_layer_discrepancy(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    # Per-sample mean absolute difference between two KxK relation matrices.
    return (r1 - r2).abs().mean(dim=(1, 2))


def relation_discrepancy(
    rels1: List[torch.Tensor],
    rels2: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Aggregate layer-wise relation differences used in Equations (4) and (14)."""
    assert len(rels1) == len(rels2)
    n = len(rels1)
    if layer_weights is None:
        layer_weights = [1.0] * n
    total = 0.0
    ws = 0.0
    for w, r1, r2 in zip(layer_weights, rels1, rels2):
        total = total + float(w) * _relation_layer_discrepancy(r1, r2)
        ws += float(w)
    return total / max(ws, 1e-8)


def relation_entropy_confidence(
    rels: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relation confidence as one minus normalized row entropy (Equation (7))."""
    n = len(rels)
    if layer_weights is None:
        layer_weights = [1.0] * n
    total = 0.0
    ws = 0.0
    for w, r in zip(layer_weights, rels):
        K = r.shape[-1]
        ent = -(r.clamp_min(eps) * torch.log(r.clamp_min(eps))).sum(dim=-1) / math.log(K + 1e-12)
        conf = (1.0 - ent.mean(dim=-1)).clamp(0.0, 1.0)
        total = total + float(w) * conf
        ws += float(w)
    return total / max(ws, 1e-8)


def relation_kl_loss_per_sample(
    student_rels: List[torch.Tensor],
    teacher_rels: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample relation distillation KL used by Equations (12) and (15)."""
    assert len(student_rels) == len(teacher_rels)
    n = len(student_rels)
    if layer_weights is None:
        layer_weights = [1.0] * n
    total = 0.0
    ws = 0.0
    for w, rs, rt in zip(layer_weights, student_rels, teacher_rels):
        rt = rt.detach().clamp_min(eps)
        rs = rs.clamp_min(eps)
        kl = (rt * (torch.log(rt) - torch.log(rs))).sum(dim=-1).mean(dim=-1)  # B
        total = total + float(w) * kl
        ws += float(w)
    return total / max(ws, 1e-8)


def relation_kl_loss(
    student_rels: List[torch.Tensor],
    teacher_rels: List[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    layer_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    loss_ps = relation_kl_loss_per_sample(student_rels, teacher_rels, layer_weights=layer_weights)
    return safe_mean(loss_ps, weight)


def normalize_batch_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min--max normalize one mini-batch of discrepancy values to [0, 1]."""
    if x.numel() <= 1:
        return torch.zeros_like(x)
    xmin = x.detach().min()
    xmax = x.detach().max()
    return ((x - xmin) / (xmax - xmin + eps)).clamp(0.0, 1.0)


@torch.no_grad()
def build_reliable_masks(
    d_feat: torch.Tensor,
    d_rel: torch.Tensor,
    reliability: torch.Tensor,
    low_q: float = 0.35,
    high_q: float = 0.75,
    min_reliability_q: float = 0.35,
    min_unchanged_ratio: float = 0.15,
    max_changed_ratio: float = 0.18,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build reliability-protected unchanged and changed masks without ground truth.

    This function implements the practical form of Equations (9)--(11). Feature and
    relation discrepancies are normalized and averaged into a joint cue. Quantile rules
    then construct non-overlapping pseudo-anchor sets:

    - Strong unchanged anchors: low joint discrepancy + reliable relation.
    - Weak unchanged anchors: bottom `min_unchanged_ratio` discrepancy samples, regardless of
      relation confidence. This prevents the unchanged set from disappearing when relation
      entropy confidence is under-calibrated in early DINO-free training.
    - Changed anchors: high discrepancy + reliable relation, capped by `max_changed_ratio` to
      avoid the training objective becoming change-amplification only.
    - Changed anchors never overlap unchanged anchors.
    """
    d_joint = 0.5 * normalize_batch_01(d_feat) + 0.5 * normalize_batch_01(d_rel)
    n = int(d_joint.numel())
    if n <= 2:
        med = d_joint.median()
        rel_thr = reliability.median()
        rel_ok = reliability >= rel_thr
        mu = (d_joint <= med).float()
        mc = ((d_joint > med) & rel_ok).float()
        mc = mc * (1.0 - mu)
        return mu, mc, d_joint, rel_ok.float()

    low_q = float(low_q)
    high_q = float(high_q)
    min_unchanged_ratio = float(max(0.01, min(min_unchanged_ratio, 0.50)))
    max_changed_ratio = float(max(0.01, min(max_changed_ratio, 0.50)))

    low_thr = torch.quantile(d_joint.detach(), low_q)
    weak_u_thr = torch.quantile(d_joint.detach(), min_unchanged_ratio)

    # Use the stricter of user high_q and the cap implied by max_changed_ratio.
    changed_q = max(high_q, 1.0 - max_changed_ratio)
    high_thr = torch.quantile(d_joint.detach(), changed_q)

    rel_thr = torch.quantile(reliability.detach(), float(min_reliability_q))
    rel_ok = reliability >= rel_thr

    strong_u = (d_joint <= low_thr) & rel_ok
    weak_u = d_joint <= weak_u_thr
    mu = (strong_u | weak_u).float()

    mc = ((d_joint >= high_thr) & rel_ok).float()
    if float(mc.sum().detach().cpu()) < 1.0:
        mc = (d_joint >= high_thr).float()

    # Unchanged anchors have priority. Do not let changed anchors overlap U.
    mc = mc * (1.0 - mu)
    return mu, mc, d_joint, rel_ok.float()


def change_preserving_relation_loss(
    d_rel_student: torch.Tensor,
    changed_mask: torch.Tensor,
    margin: float = 0.08,
) -> torch.Tensor:
    """Apply the margin loss in Equation (13) on reliable changed anchors."""
    return safe_mean(F.relu(float(margin) - d_rel_student), changed_mask)


def compact_relation_diversity_loss(
    tokens_list: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Prevent compact tokens from collapsing to identical vectors.

    Equations (16)--(17) penalize squared off-diagonal token affinities while leaving
    self-similarity on the diagonal unpenalized.
    """
    n = len(tokens_list)
    if layer_weights is None:
        layer_weights = [1.0] * n
    total = 0.0
    ws = 0.0
    for w, z in zip(layer_weights, tokens_list):
        K = z.shape[1]
        sim = torch.bmm(z, z.transpose(1, 2))
        eye = torch.eye(K, device=z.device, dtype=z.dtype).unsqueeze(0)
        off = (sim * (1.0 - eye)).pow(2).mean(dim=(1, 2))
        total = total + float(w) * off.mean()
        ws += float(w)
    return total / max(ws, 1e-8)
