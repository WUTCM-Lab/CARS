"""
Training and evaluation entry point for CARS.

The script follows the method pipeline described in the paper:

1. Sample paired local patches and create two augmented views per temporal image.
2. Use the EMA teacher to produce relation targets, feature/relation discrepancies,
   relation-entropy confidence, and reliability-protected pseudo-anchors.
3. Optimize same-modal relation consistency, reliable unchanged structural alignment,
   change-preserving relation discrepancy, weak feature anchors, and compact-token
   diversity under conflict-aware objective weights.
4. Update the teacher from the student by exponential moving average after every step.
5. At inference, discard the teacher and use Adaptive Relation-Aware Change Scoring
   (ARCS) to fuse feature and relation discrepancies from unlabeled score statistics.

Ground-truth maps are loaded only for metric computation and TP/FP/FN visualization;
the training loop explicitly discards the label returned by the data loader.

Example:
    python train_CARS_relation_ARCS.py --data_name yellow \
        --t1_path ./data/Yellow/yellow_C_1.bmp \
        --t2_path ./data/Yellow/yellow_C_2.bmp \
        --gt_path ./data/Yellow/yellow_C_gt.bmp
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import scipy.io as io
from sklearn import metrics
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader

from utils.data import Data_Loader, to_normalization
from utils.Evaluation import Evaluation
from model.CARS import (
    CARSStudentTeacher,
    build_reliable_masks,
    change_preserving_relation_loss,
    compact_relation_diversity_loss,
    feature_discrepancy,
    relation_discrepancy,
    relation_entropy_confidence,
    relation_kl_loss,
    safe_mean,
    normalize_batch_01,
)


# -------------------------
# General utilities
# -------------------------
def ensure_dir(path: str):
    """Create an output directory without failing when it already exists."""
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch and request deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Derive deterministic NumPy/Python seeds for each data-loader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def minmax01(arr: np.ndarray) -> np.ndarray:
    """Normalize a NumPy score vector or map to [0, 1] with a constant-map guard."""
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def to_uint8_01(arr01: np.ndarray) -> np.ndarray:
    return (np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8)


def threshold_cmi(cmi_u8: np.ndarray, method: str, fixed_thr: float) -> np.ndarray:
    """Convert a continuous change-intensity map into a binary change map."""
    if method == "otsu":
        _, bcm = cv2.threshold(cmi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bcm.astype(np.uint8)
    return (cmi_u8 > float(fixed_thr)).astype(np.uint8) * 255


def to_3ch_image(img: np.ndarray, fallback_gray: np.ndarray) -> np.ndarray:
    if img is None:
        return np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    return np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)


def overlay_tp_fp_fn(base_bgr: np.ndarray, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    """Overlay true positives (green), false positives (red), and false negatives (blue)."""
    out = base_bgr.copy()
    tp_layer = np.zeros_like(out)
    fp_layer = np.zeros_like(out)
    fn_layer = np.zeros_like(out)
    tp_layer[tp > 0] = (0, 255, 0)
    fp_layer[fp > 0] = (0, 0, 255)
    fn_layer[fn > 0] = (255, 0, 0)
    out = cv2.addWeighted(out, 1.0, tp_layer, 0.45, 0)
    out = cv2.addWeighted(out, 1.0, fp_layer, 0.45, 0)
    out = cv2.addWeighted(out, 1.0, fn_layer, 0.45, 0)
    return out


def match_channels(x: torch.Tensor, target_c: int) -> torch.Tensor:
    """Conservative channel fix for shared-backbone training."""
    c = x.shape[1]
    if c == target_c:
        return x
    if c == 1 and target_c == 3:
        return x.repeat(1, 3, 1, 1)
    if c > target_c:
        return x[:, :target_c]
    # pad by repeating the last channel
    repeat = target_c - c
    return torch.cat([x, x[:, -1:].repeat(1, repeat, 1, 1)], dim=1)


def build_aug():
    """Create the two stochastic views used for teacher--student self-distillation."""
    strong = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.35, 0.35, 0.35, 0.08),
        T.RandomGrayscale(p=0.15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ])
    weak = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.15, 0.15, 0.15, 0.03),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    return strong, weak


def apply_aug_batch(batch: torch.Tensor, aug) -> torch.Tensor:
    """Apply a torchvision transform independently to every patch in a mini-batch."""
    return torch.stack([aug(img) for img in batch], dim=0)


# -------------------------
# Adaptive relation-aware scoring
# -------------------------
class AdaptiveRelationScorer:
    """
    Label-free, epoch-wise fusion of feature and relation discrepancy (ARCS).

    The update never uses the ground-truth map.  At each validation epoch it estimates
    the reliability of each cue from its normalized discrepancy distribution:

      1) separation quality between low- and high-discrepancy quantiles;
      2) global dynamic range;
      3) EMA-teacher relation quality for the relation cue;
      4) feature--relation conflict, used as a conservative gate on relation evidence.

    The proposed relation weight is clipped to a safe interval and then smoothed with
    exponential moving average.  Consequently, the fusion coefficients can adapt to
    dataset characteristics while gradually converging instead of oscillating between
    epochs.
    """

    def __init__(
        self,
        init_feat_weight: float = 0.90,
        momentum: float = 0.90,
        min_rel_weight: float = 0.05,
        max_rel_weight: float = 0.35,
        temperature: float = 0.35,
        low_quantile: float = 0.20,
        high_quantile: float = 0.80,
        eps: float = 1e-8,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"adaptive_score_momentum must be in [0, 1), got {momentum}")
        if not 0.0 <= min_rel_weight <= max_rel_weight <= 1.0:
            raise ValueError(
                "Adaptive relation-weight bounds must satisfy "
                f"0 <= min <= max <= 1, got {min_rel_weight}, {max_rel_weight}"
            )
        if not 0.0 < temperature:
            raise ValueError(f"adaptive_score_temperature must be positive, got {temperature}")
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError(
                "Adaptive score quantiles must satisfy 0 <= low < high <= 1, got "
                f"{low_quantile}, {high_quantile}"
            )

        init_rel_weight = float(np.clip(1.0 - init_feat_weight, min_rel_weight, max_rel_weight))
        self.w_rel = init_rel_weight
        self.w_feat = 1.0 - init_rel_weight
        self.momentum = float(momentum)
        self.min_rel_weight = float(min_rel_weight)
        self.max_rel_weight = float(max_rel_weight)
        self.temperature = float(temperature)
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        self.eps = float(eps)
        self.history: List[Dict[str, float]] = []

    def _cue_quality(self, values: np.ndarray) -> float:
        """Estimate label-free cue quality from quantile separation and dynamic range."""
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size < 4:
            return 0.0

        low_thr = float(np.quantile(x, self.low_quantile))
        high_thr = float(np.quantile(x, self.high_quantile))
        low_set = x[x <= low_thr]
        high_set = x[x >= high_thr]
        if low_set.size == 0 or high_set.size == 0:
            return 0.0

        gap = max(float(high_set.mean() - low_set.mean()), 0.0)
        within = 0.5 * (float(low_set.std()) + float(high_set.std()))
        separation = gap / (gap + within + self.eps)

        # A nearly constant map is not considered reliable even if its quantile gap is
        # numerically non-zero.  The dynamic term penalizes such collapsed score maps.
        spread = float(x.std())
        dynamic = spread / (spread + 0.10 + self.eps)
        quality = 0.70 * separation + 0.30 * dynamic
        return float(np.clip(quality, 0.0, 1.0))

    def update(
        self,
        feat01: np.ndarray,
        rel01: np.ndarray,
        relation_quality: float,
        epoch: int,
    ) -> Dict[str, float]:
        feat = np.asarray(feat01, dtype=np.float64).reshape(-1)
        rel = np.asarray(rel01, dtype=np.float64).reshape(-1)
        if feat.shape != rel.shape:
            raise ValueError(
                f"Feature/relation score shapes must match, got {feat.shape} and {rel.shape}"
            )

        # Estimate the two cue qualities from the target score distributions only.
        q_feat = self._cue_quality(feat)
        q_rel_base = self._cue_quality(rel)
        rho = float(np.clip(relation_quality, 0.0, 1.0))

        # Keep the relation branch trainable when entropy confidence is under-calibrated,
        # but still let a reliable teacher increase its contribution.
        q_rel = q_rel_base * (0.5 + 0.5 * rho)

        # If feature and relation maps strongly disagree, relation evidence is introduced
        # more conservatively.  This is label-free and consistent with the conflict-aware
        # principle used during training.
        score_conflict = float(np.mean(np.abs(feat - rel)))
        conflict_gate = float(np.clip(1.0 - 0.65 * score_conflict, 0.35, 1.0))
        q_rel *= conflict_gate

        # Map the two quality estimates to normalized fusion weights, corresponding to
        # the label-free mapping Phi(.) in Equation (23), then enforce the safe interval.
        logits = np.asarray([q_feat, q_rel], dtype=np.float64) / self.temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + self.eps

        proposed_rel = float(np.clip(probs[1], self.min_rel_weight, self.max_rel_weight))
        proposed_feat = 1.0 - proposed_rel

        # Equation (24): smooth the proposed relation weight across epochs by EMA.
        self.w_rel = float(
            np.clip(
                self.momentum * self.w_rel + (1.0 - self.momentum) * proposed_rel,
                self.min_rel_weight,
                self.max_rel_weight,
            )
        )
        self.w_feat = 1.0 - self.w_rel

        state = {
            "score_epoch": float(epoch),
            "score_feat_weight": float(self.w_feat),
            "score_rel_weight": float(self.w_rel),
            "score_proposed_feat_weight": float(proposed_feat),
            "score_proposed_rel_weight": float(proposed_rel),
            "score_q_feat": float(q_feat),
            "score_q_rel": float(q_rel),
            "score_relation_quality": float(rho),
            "score_conflict": float(score_conflict),
            "score_conflict_gate": float(conflict_gate),
        }
        self.history.append(state)
        return state

    def get_weights(self) -> Tuple[float, float]:
        return float(self.w_feat), float(self.w_rel)

    def state_dict(self) -> Dict[str, object]:
        return {
            "w_feat": float(self.w_feat),
            "w_rel": float(self.w_rel),
            "momentum": float(self.momentum),
            "min_rel_weight": float(self.min_rel_weight),
            "max_rel_weight": float(self.max_rel_weight),
            "temperature": float(self.temperature),
            "low_quantile": float(self.low_quantile),
            "high_quantile": float(self.high_quantile),
            "history": self.history,
        }


# -------------------------
# Conflict-aware weighting
# -------------------------
def conflict_aware_weights(
    epoch: int,
    args,
    reliability: torch.Tensor,
    d_feat: torch.Tensor,
    d_rel: torch.Tensor,
    mask_u: torch.Tensor,
    mask_c: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute the practical conflict-aware weights used in Equations (18)--(21).

    The previous CARS version over-trusted relation entropy confidence. When relation
    matrices were still close to uniform, relQ collapsed to ~0 and permanently disabled
    reliable unchanged alignment. This version keeps the diagnostic raw relQ, but uses a
    safe floor and an explicit alignment floor after warm-up so the second framework
    module actually trains.
    """
    rel_quality_raw = float(reliability.detach().mean().clamp(0.0, 1.0).cpu())
    rel_quality = float(max(rel_quality_raw, float(args.relq_floor)))

    nf = normalize_batch_01(d_feat.detach())
    nr = normalize_batch_01(d_rel.detach())
    conflict = float((nf - nr).abs().mean().clamp(0.0, 1.0).cpu())
    u_ratio = float(mask_u.detach().mean().cpu())
    c_ratio = float(mask_c.detach().mean().cpu())

    # No warm-up schedule: all objectives are active from the first epoch.
    # The conflict-aware terms below still adapt weights according to relation reliability,
    # feature-relation conflict, and anchor availability.
    sm_ramp = 1.0
    align_ramp = 1.0
    chg_ramp = 1.0
    div_ramp = 1.0

    # Same-modal relation consistency remains the base stabilizer.
    w_sm = args.w_rel_sm * sm_ramp * (0.5 + 0.5 * rel_quality)

    # Reliable unchanged structural alignment must not disappear. The data-dependent term
    # still reacts to conflict and U availability, but an alignment floor keeps the loss
    # active from the first epoch.
    align_availability = max(u_ratio, 0.5 * float(args.min_unchanged_ratio))
    conflict_gate = max(0.25, 1.0 - 0.50 * conflict)
    w_align_data = args.w_rel_align * align_ramp * (0.5 + 0.5 * rel_quality) * conflict_gate * align_availability
    w_align_floor = args.align_floor * align_ramp
    w_align = max(w_align_data, w_align_floor)

    # Change preservation is useful only when alignment exists. Cap it relative to alignment
    # to avoid the late-stage behaviour where the model only amplifies differences.
    w_change_data = args.w_change * chg_ramp * (0.5 + 0.5 * rel_quality) * max(c_ratio, 0.03) * (0.35 + 0.65 * conflict)
    w_change_cap = max(args.min_change_weight, args.change_align_cap * max(w_align, 1e-8))
    w_change = min(w_change_data, w_change_cap)

    # Diversity is weak but should not be zero, otherwise relation tokens collapse.
    w_compact = args.w_compact * div_ramp * (0.5 + 0.5 * rel_quality)

    return {
        "w_sm": float(max(w_sm, 0.0)),
        "w_align": float(max(w_align, 0.0)),
        "w_change": float(max(w_change, 0.0)),
        "w_compact": float(max(w_compact, 0.0)),
        "rel_quality": rel_quality_raw,
        "rel_quality_safe": rel_quality,
        "conflict": conflict,
        "u_ratio": u_ratio,
        "c_ratio": c_ratio,
    }


# -------------------------
# Training
# -------------------------
def train_one_epoch(model, loader, optimizer, device, epoch: int, args, layer_weights: List[float]):
    """Run one complete CARS optimization epoch over unlabeled paired patches."""
    model.train()
    aug1, aug2 = build_aug()

    running = {
        "loss": 0.0,
        "rel_sm": 0.0,
        "rel_align": 0.0,
        "change": 0.0,
        "compact": 0.0,
        "rel_quality": 0.0,
        "rel_quality_safe": 0.0,
        "feat_sm": 0.0,
        "feat_align": 0.0,
        "conflict": 0.0,
        "u_ratio": 0.0,
        "c_ratio": 0.0,
        "w_sm": 0.0,
        "w_align": 0.0,
        "w_change": 0.0,
        "w_compact": 0.0,
    }

    desc = f"Train [CARS] [Epoch {epoch}]"
    for t1, t2, _ in tqdm(loader, desc=desc, ncols=170):
        # The third loader item is the center-pixel ground truth, intentionally ignored here.
        # Training, anchor mining, and objective weighting therefore remain label-free.
        t1 = match_channels(t1.float(), args.in_channels)
        t2 = match_channels(t2.float(), args.in_channels)
        t1_v1 = apply_aug_batch(t1, aug1).to(device, non_blocking=True)
        t1_v2 = apply_aug_batch(t1, aug2).to(device, non_blocking=True)
        t2_v1 = apply_aug_batch(t2, aug1).to(device, non_blocking=True)
        t2_v2 = apply_aug_batch(t2, aug2).to(device, non_blocking=True)

        with torch.no_grad():
            # Teacher-side cues define the reliable anchor sets. The teacher is frozen with
            # respect to back-propagation, making these targets more stable than the student.
            tt1 = model.teacher_forward(t1_v1)
            tt2 = model.teacher_forward(t2_v1)
            d_feat_t = feature_discrepancy(tt1["pooled"], tt2["pooled"])
            d_rel_t = relation_discrepancy(tt1["relations"], tt2["relations"], layer_weights=layer_weights)
            conf_t1 = relation_entropy_confidence(tt1["relations"], layer_weights=layer_weights)
            conf_t2 = relation_entropy_confidence(tt2["relations"], layer_weights=layer_weights)
            reliability = (conf_t1 * conf_t2).sqrt().clamp(0.0, 1.0)
            mask_u, mask_c, _, _ = build_reliable_masks(
                d_feat=d_feat_t,
                d_rel=d_rel_t,
                reliability=reliability,
                low_q=args.mask_low_q,
                high_q=args.mask_high_q,
                min_reliability_q=args.mask_reliability_q,
                min_unchanged_ratio=args.min_unchanged_ratio,
                max_changed_ratio=args.max_changed_ratio,
            )
            weights = conflict_aware_weights(epoch, args, reliability, d_feat_t, d_rel_t, mask_u, mask_c)

        # The student receives the complementary augmented views and is optimized against
        # the stop-gradient teacher targets and the cross-temporal structural constraints.
        st1 = model.student_forward(t1_v2)
        st2 = model.student_forward(t2_v2)

        # 1) Same-modal multi-level structural relation consistency.
        # A small feature-anchor term prevents the relation branch from satisfying the loss
        # with nearly uniform relation matrices. It is still DINO-free: no prototype labels,
        # no token CE, and no class-distribution matching.
        loss_rel_sm_only = 0.5 * (
            relation_kl_loss(st1["relations"], tt1["relations"], layer_weights=layer_weights) +
            relation_kl_loss(st2["relations"], tt2["relations"], layer_weights=layer_weights)
        )
        loss_feat_sm = 0.5 * (
            feature_discrepancy(st1["pooled"], tt1["pooled"].detach()).mean() +
            feature_discrepancy(st2["pooled"], tt2["pooled"].detach()).mean()
        )
        loss_rel_sm = loss_rel_sm_only + args.feat_sm_ratio * loss_feat_sm

        # 2) Reliable unchanged structural alignment. No full-image blind cross-modal alignment.
        # The relation term is the main loss; the weak feature anchor makes the unchanged
        # anchors numerically effective even when early relation distributions are flat.
        loss_align_rel = 0.5 * (
            relation_kl_loss(st1["relations"], tt2["relations"], weight=mask_u, layer_weights=layer_weights) +
            relation_kl_loss(st2["relations"], tt1["relations"], weight=mask_u, layer_weights=layer_weights)
        )
        loss_feat_align = 0.5 * (
            safe_mean(feature_discrepancy(st1["pooled"], tt2["pooled"].detach()), mask_u) +
            safe_mean(feature_discrepancy(st2["pooled"], tt1["pooled"].detach()), mask_u)
        )
        loss_align = loss_align_rel + args.feat_align_ratio * loss_feat_align

        # 3) Change-preserving relation discrepancy on reliable changed pairs.
        d_rel_s = relation_discrepancy(st1["relations"], st2["relations"], layer_weights=layer_weights)
        loss_change = change_preserving_relation_loss(d_rel_s, mask_c, margin=args.change_margin)

        # Weak diversity regularization to avoid compact relation-token collapse.
        loss_compact = 0.5 * (
            compact_relation_diversity_loss(st1["tokens"], layer_weights=layer_weights) +
            compact_relation_diversity_loss(st2["tokens"], layer_weights=layer_weights)
        )

        # Equation (21): combine the main alignment objective, complementary change term,
        # stabilization term, and representation constraint with conflict-aware weights.
        loss = (
            weights["w_sm"] * loss_rel_sm +
            weights["w_align"] * loss_align +
            weights["w_change"] * loss_change +
            weights["w_compact"] * loss_compact
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            # Update the target network only after the student optimization step.
            model.update_teacher(args.teacher_m)

        running["loss"] += float(loss.detach().cpu())
        running["rel_sm"] += float(loss_rel_sm.detach().cpu())
        running["rel_align"] += float(loss_align.detach().cpu())
        running["change"] += float(loss_change.detach().cpu())
        running["compact"] += float(loss_compact.detach().cpu())
        running["feat_sm"] += float(loss_feat_sm.detach().cpu())
        running["feat_align"] += float(loss_feat_align.detach().cpu())
        for k in ["rel_quality", "rel_quality_safe", "conflict", "u_ratio", "c_ratio", "w_sm", "w_align", "w_change", "w_compact"]:
            running[k] += weights[k]

    denom = max(1, len(loader))
    return {k: v / denom for k, v in running.items()}


# -------------------------
# Validation / relation-aware scoring
# -------------------------
@torch.no_grad()
def validate_change_map(
    model,
    loader,
    o_h,
    o_w,
    gt,
    vis_dir,
    device,
    args,
    adaptive_scorer: AdaptiveRelationScorer,
    epoch: int,
    relation_quality: float,
    x1_full=None,
    x2_full=None,
):
    """Run student-only inference, ARCS fusion, thresholding, and evaluation."""
    model.eval()
    feat_scores, rel_scores = [], []

    # Test mode extracts one centered patch per image pixel, so the flattened score order
    # can be reshaped directly to the original HxW image after all batches are processed.
    for t1, t2, _ in tqdm(loader, desc="Validate [CARS]", ncols=140, colour="cyan"):
        t1 = match_channels(t1.float(), args.in_channels).to(device, non_blocking=True)
        t2 = match_channels(t2.float(), args.in_channels).to(device, non_blocking=True)
        out = model.inference_pair(t1, t2)
        feat_scores.extend(out["d_feat"].cpu().numpy().tolist())
        rel_scores.extend(out["d_rel"].cpu().numpy().tolist())

    feat_arr = np.array(feat_scores, dtype=np.float32)
    rel_arr = np.array(rel_scores, dtype=np.float32)
    feat01 = minmax01(feat_arr)
    rel01 = minmax01(rel_arr)

    # 5) Adaptive relation-aware change scoring (ARCS).
    # The adaptive update uses only unlabeled discrepancy statistics and the current
    # relation quality; GT is used below only for evaluation.
    if args.score_mode == "adaptive":
        score_state = adaptive_scorer.update(
            feat01=feat01,
            rel01=rel01,
            relation_quality=relation_quality,
            epoch=epoch,
        )
        score_feat_weight = score_state["score_feat_weight"]
        score_rel_weight = score_state["score_rel_weight"]
    else:
        weight_sum = float(args.score_feat_weight + args.score_rel_weight)
        if weight_sum <= 0.0:
            raise ValueError("Fixed scoring weights must have a positive sum.")
        score_feat_weight = float(args.score_feat_weight / weight_sum)
        score_rel_weight = float(args.score_rel_weight / weight_sum)
        score_state = {
            "score_epoch": float(epoch),
            "score_feat_weight": score_feat_weight,
            "score_rel_weight": score_rel_weight,
            "score_proposed_feat_weight": score_feat_weight,
            "score_proposed_rel_weight": score_rel_weight,
            "score_q_feat": 0.0,
            "score_q_rel": 0.0,
            "score_relation_quality": float(relation_quality),
            "score_conflict": float(np.mean(np.abs(feat01 - rel01))),
            "score_conflict_gate": 1.0,
        }

    # Equation (22): fuse the normalized feature and relation discrepancy cues.
    score = score_feat_weight * feat01 + score_rel_weight * rel01
    score01 = minmax01(score).reshape(o_h, o_w)
    feat_map = feat01.reshape(o_h, o_w)
    rel_map = rel01.reshape(o_h, o_w)

    cmi_u8 = to_uint8_01(score01)
    feat_u8 = to_uint8_01(feat_map)
    rel_u8 = to_uint8_01(rel_map)
    cv2.imwrite(os.path.join(vis_dir, "CMI_CARS_relation_aware.png"), cmi_u8)
    cv2.imwrite(os.path.join(vis_dir, "CMI_feature_discrepancy.png"), feat_u8)
    cv2.imwrite(os.path.join(vis_dir, "CMI_relation_discrepancy.png"), rel_u8)

    bcm = threshold_cmi(cmi_u8, args.threshold_method, args.fixed_thr)
    cv2.imwrite(os.path.join(vis_dir, "BCM_CARS.png"), bcm)

    # Ground truth enters only at this point, after the score map and binary prediction
    # have already been produced, and is used solely for reporting evaluation metrics.
    evaler = Evaluation(gt.astype(np.uint8), bcm)
    OA, KC, _ = evaler.Classification_indicators()
    _, _, F1 = evaler.ObjectExtract_indicators()
    FPR, TPR, _ = metrics.roc_curve((gt > 127).astype(int).flatten(), score01.flatten())
    AUC = metrics.auc(FPR, TPR)

    pred = (bcm > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    FP = ((pred == 1) & (gt_bin == 0)).astype(np.uint8) * 255
    FN = ((pred == 0) & (gt_bin == 1)).astype(np.uint8) * 255
    TP = ((pred == 1) & (gt_bin == 1)).astype(np.uint8) * 255
    base = x2_full if x2_full is not None else x1_full
    base_bgr = to_3ch_image(base, cmi_u8)
    if base_bgr.shape[:2] != TP.shape[:2]:
        H, W = base_bgr.shape[:2]
        TP = cv2.resize(TP, (W, H), interpolation=cv2.INTER_NEAREST)
        FP = cv2.resize(FP, (W, H), interpolation=cv2.INTER_NEAREST)
        FN = cv2.resize(FN, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_FP_FN.png"), overlay_tp_fp_fn(base_bgr, TP, FP, FN))

    metrics_dict = {
        "AUC": float(AUC),
        "OA": float(OA),
        "KC": float(KC),
        "F1": float(F1),
        **{k: float(v) for k, v in score_state.items()},
    }
    print(
        f"[VAL][CARS-ARCS] AUC={AUC:.4f} OA={OA:.2f} KC={KC:.2f} F1={F1:.2f} "
        f"score_w={score_feat_weight:.4f}/{score_rel_weight:.4f} "
        f"q={score_state['score_q_feat']:.4f}/{score_state['score_q_rel']:.4f} "
        f"score_conflict={score_state['score_conflict']:.4f}"
    )
    return cmi_u8, bcm, metrics_dict, {
        "feat_map": feat_u8,
        "rel_map": rel_u8,
        "score_feat_weight": np.asarray([[score_feat_weight]], dtype=np.float32),
        "score_rel_weight": np.asarray([[score_rel_weight]], dtype=np.float32),
    }


# -------------------------
# Main
# -------------------------
def build_argparser():
    """Define dataset, model, optimization, anchor-mining, and ARCS options."""
    parser = argparse.ArgumentParser("CARS relation-distillation training")
    parser.add_argument("--data_name", default="yellow", type=str)
    parser.add_argument("--t1_path", default="./data/Yellow/yellow_C_1.bmp", type=str)
    parser.add_argument("--t2_path", default="./data/Yellow/yellow_C_2.bmp", type=str)
    parser.add_argument("--gt_path", default="./data/Yellow/yellow_C_gt.bmp", type=str)
    parser.add_argument("--t1_nc", default=3, type=int)
    parser.add_argument("--t2_nc", default=3, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--patch_size", default=11, type=int)
    parser.add_argument("--test_ps", default=11, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--vision_path", default="./dino_vision/CARS_relation_NoWarmUp/", type=str)

    # Model
    parser.add_argument("--feat_dim", default=64, type=int)
    parser.add_argument("--rel_dim", default=128, type=int)
    parser.add_argument("--num_relation_tokens", default=16, type=int)
    parser.add_argument("--attn_temp", default=0.20, type=float)
    parser.add_argument("--relation_temp", default=0.20, type=float)
    parser.add_argument("--teacher_m", default=0.996, type=float)

    # Reliable masks
    parser.add_argument("--mask_low_q", default=0.35, type=float)
    parser.add_argument("--mask_high_q", default=0.75, type=float)
    parser.add_argument("--mask_reliability_q", default=0.35, type=float)
    parser.add_argument("--min_unchanged_ratio", default=0.15, type=float,
                        help="Minimum weak unchanged anchor ratio. Keeps U from disappearing.")
    parser.add_argument("--max_changed_ratio", default=0.18, type=float,
                        help="Cap changed anchors to avoid change-only optimization.")

    # Loss weights: actual values will be conflict-aware adjusted.
    parser.add_argument("--w_rel_sm", default=1.00, type=float)
    parser.add_argument("--w_rel_align", default=0.60, type=float)
    parser.add_argument("--w_change", default=0.10, type=float)
    parser.add_argument("--w_compact", default=0.005, type=float)
    parser.add_argument("--change_margin", default=0.03, type=float)
    parser.add_argument("--relq_floor", default=0.05, type=float,
                        help="Safe floor for relation quality used by CAW. Raw relQ is still logged.")
    parser.add_argument("--align_floor", default=0.03, type=float,
                        help="Warm-up floor for reliable unchanged structural alignment.")
    parser.add_argument("--change_align_cap", default=0.60, type=float,
                        help="Upper bound of change weight relative to active alignment weight.")
    parser.add_argument("--min_change_weight", default=0.0, type=float)
    parser.add_argument("--feat_sm_ratio", default=0.05, type=float,
                        help="Weak same-modal feature anchor ratio; DINO-free.")
    parser.add_argument("--feat_align_ratio", default=0.20, type=float,
                        help="Weak reliable-unchanged feature alignment anchor ratio; DINO-free.")
    parser.add_argument("--grad_clip", default=5.0, type=float)

    # Adaptive relation-aware change scoring.
    # Fixed mode is kept only for ablation and backward-compatible evaluation.
    parser.add_argument("--score_mode", default="adaptive", choices=["adaptive", "fixed"], type=str)
    parser.add_argument("--score_feat_weight", default=0.90, type=float,
                        help="Feature weight used only when --score_mode fixed.")
    parser.add_argument("--score_rel_weight", default=0.10, type=float,
                        help="Relation weight used only when --score_mode fixed.")
    parser.add_argument("--adaptive_init_feat_weight", default=0.90, type=float)
    parser.add_argument("--adaptive_score_momentum", default=0.90, type=float,
                        help="EMA momentum for epoch-wise score-weight updates.")
    parser.add_argument("--adaptive_min_rel_weight", default=0.05, type=float)
    parser.add_argument("--adaptive_max_rel_weight", default=0.35, type=float)
    parser.add_argument("--adaptive_score_temperature", default=0.35, type=float)
    parser.add_argument("--adaptive_score_low_q", default=0.20, type=float)
    parser.add_argument("--adaptive_score_high_q", default=0.80, type=float)
    parser.add_argument("--threshold_method", default="otsu", choices=["fixed", "otsu"], type=str)
    parser.add_argument("--fixed_thr", default=66.3, type=float)
    return parser


def resolve_input_channels(args):
    """Select the shared-backbone channel count used for both temporal modalities."""
    if args.in_channels is not None:
        return int(args.in_channels)
    return int(max(args.t1_nc, args.t2_nc))


def load_full_images_and_gt(args):
    """Load full-resolution images for visualization and GT for evaluation only."""
    if args.data_name == "bastrop":
        mat = io.loadmat(args.t1_path)
        x1 = mat["t1_L5"][:, :, 3]
        x2 = mat["t2_ALI"][:, :, 5]
        x1 = to_normalization(x1)[..., np.newaxis]
        x2 = to_normalization(x2)[..., np.newaxis]
        gt = (mat["ROI_1"] * 255).astype(np.uint8)
        x1_full = np.repeat((x1 * 255).astype(np.uint8), 3, axis=2)
        x2_full = np.repeat((x2 * 255).astype(np.uint8), 3, axis=2)
    else:
        x1_full = cv2.imread(args.t1_path)
        x2_full = cv2.imread(args.t2_path)
        gt_img = cv2.imread(args.gt_path)
        if gt_img is None:
            raise FileNotFoundError(args.gt_path)
        gt = gt_img[:, :, 0].astype(np.uint8)
    return x1_full, x2_full, gt


def main():
    """Construct the experiment, train CARS, and save epoch-wise artifacts."""
    args = build_argparser().parse_args()
    set_seed(args.seed)
    args.in_channels = resolve_input_channels(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.vision_path, "CARS_routeB_ARCS_no_warmup")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Running CARS Route-B ARCS No-Warmup: Adaptive Relation Self-Distillation")
    print("DINO-free: no prototype head, no online token CE, no blind full-image CM distillation")
    print(vars(args))
    print("=" * 80)

    # Training samples patches sparsely with stride equal to the patch size, whereas test
    # mode enumerates every pixel to reconstruct a dense change-intensity map.
    train_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.patch_size, mode="train", transform=T.ToTensor()
    )
    test_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.test_ps, mode="test", transform=T.ToTensor()
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, worker_init_fn=seed_worker, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 6, shuffle=False,
        num_workers=0, worker_init_fn=seed_worker, generator=generator
    )

    x1_full, x2_full, gt = load_full_images_and_gt(args)
    o_h, o_w = gt.shape[:2]

    # The student and EMA teacher share the architecture but not the update rule.
    model = CARSStudentTeacher(
        in_channels=args.in_channels,
        feat_dim=args.feat_dim,
        rel_dim=args.rel_dim,
        num_relation_tokens=args.num_relation_tokens,
        attn_temp=args.attn_temp,
        relation_temp=args.relation_temp,
        teacher_momentum=args.teacher_m,
    ).to(device)
    optimizer = optim.RMSprop(
        list(model.student_backbone.parameters()) + list(model.student_relation.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
        momentum=0.9,
    )

    # ARCS keeps feature discrepancy dominant initially and learns a bounded relation
    # contribution from unlabeled score statistics during validation.
    adaptive_scorer = AdaptiveRelationScorer(
        init_feat_weight=args.adaptive_init_feat_weight,
        momentum=args.adaptive_score_momentum,
        min_rel_weight=args.adaptive_min_rel_weight,
        max_rel_weight=args.adaptive_max_rel_weight,
        temperature=args.adaptive_score_temperature,
        low_quantile=args.adaptive_score_low_q,
        high_quantile=args.adaptive_score_high_q,
    )


    layer_weights = [1, 1, 1, 1]
    history = []
    best_auc = -1.0
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, args, layer_weights)
        print(
            f"[EPOCH {epoch}][CARS] "
            f"loss={train_stats['loss']:.5f} "
            f"sm={train_stats['rel_sm']:.5f} align={train_stats['rel_align']:.5f} "
            f"chg={train_stats['change']:.5f} comp={train_stats['compact']:.5f} "
            f"featSM={train_stats['feat_sm']:.5f} featAL={train_stats['feat_align']:.5f} "
            f"relQ={train_stats['rel_quality']:.4f}/{train_stats['rel_quality_safe']:.4f} conflict={train_stats['conflict']:.4f} "
            f"U={train_stats['u_ratio']:.3f} C={train_stats['c_ratio']:.3f} "
            f"w_sm={train_stats['w_sm']:.4f} w_align={train_stats['w_align']:.4f} "
            f"w_chg={train_stats['w_change']:.4f} w_comp={train_stats['w_compact']:.4f}"
        )

        ep_dir = os.path.join(out_dir, str(epoch))
        ensure_dir(ep_dir)
        cmi_u8, bcm_u8, val_metrics, extra_maps = validate_change_map(
            model=model,
            loader=test_loader,
            o_h=o_h,
            o_w=o_w,
            gt=gt,
            vis_dir=ep_dir,
            device=device,
            args=args,
            adaptive_scorer=adaptive_scorer,
            epoch=epoch,
            relation_quality=train_stats["rel_quality_safe"],
            x1_full=x1_full,
            x2_full=x2_full,
        )
        io.savemat(
            os.path.join(ep_dir, "CARS_outputs.mat"),
            {"cmi_u8": cmi_u8, "bcm_u8": bcm_u8, **extra_maps},
        )
        record = {"epoch": epoch, **{k: float(v) for k, v in train_stats.items()}, **val_metrics}
        history.append(record)
        with open(os.path.join(ep_dir, "metrics_CARS.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "adaptive_score_history.json"), "w", encoding="utf-8") as f:
            json.dump(adaptive_scorer.state_dict(), f, indent=2, ensure_ascii=False)
        # Save full student/teacher state for reproducibility and frozen-weight transfer.
        torch.save(model.state_dict(), os.path.join(ep_dir, f"CARS_epoch{epoch}.pth"))

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best_auc_CARS.pth"))
        if val_metrics["F1"] > best_f1:
            best_f1 = val_metrics["F1"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best_f1_CARS.pth"))

    with open(os.path.join(out_dir, "history_all.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    final_wf, final_wr = adaptive_scorer.get_weights()
    print(
        f"Training finished. Final adaptive score weights: "
        f"feature={final_wf:.4f}, relation={final_wr:.4f}"
    )


if __name__ == "__main__":
    main()
