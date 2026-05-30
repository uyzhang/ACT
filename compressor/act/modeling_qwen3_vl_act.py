"""ACT — Anchor-Centered video Token compression for Qwen3-VL.

This module patches the Qwen3-VL vision tower to compress visual tokens around
visual anchors. Three anchor-centered components run inside the vision encoder
and are then handed off to the language model:

(Q1) Anchor budget allocation. A region-level multi-marginal Sinkhorn estimates
     cross-frame change (second-order curvature on triplets of consecutive
     frames) and turns it into a per-frame anchor budget k_t whose total equals
     ``T * tokens_per_frame * retain_ratio``.

(Q2) Coverage-aware anchor selection. Within each frame we solve a weighted
     facility-location problem with a coverage-aware greedy algorithm: anchors
     are picked one by one to maximise marginal coverage of the dense token
     field, gated by saliency through a candidate pre-pool of size
     ``k * GREEDY_PRE_RATIO``.

(Q3) Residual evidence aggregation. Each non-anchor token is assigned to its
     nearest anchor (winner-takes-all, optionally constrained to the same
     spatial region) and contributes a similarity-weighted residual back into
     that anchor's feature. Token count is unchanged; information density is
     increased. The same pushforward is applied consistently to every
     deepstack feature so that all transformer scales stay aligned.
"""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLVisionModel,
)

from .curvature import compute_mm_ot_region_curvature


# =============================================================================
# Core ACT hyper-parameters
# =============================================================================

# Global keep ratio. Total anchor budget = T * tokens_per_frame * RETAIN_RATIO.
DEFAULT_RETAIN_RATIO = float(os.environ.get("ACT_RETAIN_RATIO", "0.25"))
# Per-frame minimum anchor count. Even frames with very low curvature receive
# at least MIN_K anchors so the language model never sees a fully-empty frame.
DEFAULT_MIN_K = int(os.environ.get("ACT_MIN_K", "1"))
# Softmax temperature used when distributing the global budget across frames
# according to frame-level curvature. Lower → sharper allocation toward
# high-curvature frames; higher → more uniform.
DEFAULT_BUDGET_TEMP = float(os.environ.get("ACT_BUDGET_TEMP", "0.7"))

# Region grid. Each frame's H×W token grid is partitioned into
# NUM_REGIONS_H × NUM_REGIONS_W regions. The multi-marginal Sinkhorn operates
# on regions (not raw tokens) and the same-region constraint in (Q3) is
# enforced over this grid. 4×4 keeps the OT cost tensor [16,16,16] tiny while
# providing enough spatial granularity.
DEFAULT_NUM_REGIONS_H = int(os.environ.get("ACT_NUM_REGIONS_H", "4"))
DEFAULT_NUM_REGIONS_W = int(os.environ.get("ACT_NUM_REGIONS_W", "4"))
# Region-curvature weight in token saliency:
#     score = saliency * (1 + β * region_curv_norm)
DEFAULT_CURV_WEIGHT_BETA = float(os.environ.get("ACT_CURV_WEIGHT_BETA", "1.0"))

# Multi-marginal Sinkhorn entropy regularisation and iteration count.
DEFAULT_SINKHORN_EPSILON = float(os.environ.get("ACT_SINKHORN_EPSILON", "0.05"))
DEFAULT_SINKHORN_ITERS = int(os.environ.get("ACT_SINKHORN_ITERS", "50"))

# Residual evidence aggregation strength α in
#     anchor[k*] += α * cos(d, k*) * d
# α=0 disables aggregation (pure select-and-discard); α=0.3 is the default.
DEFAULT_PROJECT_ALPHA = float(os.environ.get("ACT_PROJECT_ALPHA", "0.3"))
# Whether the pushforward is constrained to the same region.
#   1 → winner-takes-all within the same region (preserves spatial locality)
#   0 → any anchor in the frame is eligible
DEFAULT_SAME_REGION = int(os.environ.get("ACT_SAME_REGION", "1"))

# =============================================================================
# Coverage-aware anchor selection (Q2)
#
# Coverage-aware greedy is the default in the released configuration. It
# replaces in-frame top-k selection with a weighted facility-location greedy.
#
#   GREEDY_COVERAGE  = master switch
#   GREEDY_PRE_RATIO = candidate pool size multiplier. We first pick the top
#                      (k * ratio) tokens by saliency as the candidate pool
#                      and run greedy inside the pool. ratio=1 degenerates to
#                      top-k, ratio→∞ ignores saliency. 2.0 gives greedy a
#                      reasonable search space without admitting low-saliency
#                      "high-coverage" noise tokens.
# =============================================================================
DEFAULT_GREEDY_COVERAGE = int(os.environ.get("ACT_GREEDY_COVERAGE", "1"))
DEFAULT_GREEDY_PRE_RATIO = float(os.environ.get("ACT_GREEDY_PRE_RATIO", "2.0"))

# =============================================================================
# Optional knobs (off by default — kept for ablations and reproducibility)
# =============================================================================

# Adaptive curvature weight β: sigmoid-decay β based on average inter-frame
# cosine similarity. Higher inter-frame similarity (densely-sampled video) →
# noisier curvature → β automatically shrinks.
DEFAULT_ADAPTIVE_BETA = int(os.environ.get("ACT_ADAPTIVE_BETA", "0"))
DEFAULT_ABETA_CENTER = float(os.environ.get("ACT_ABETA_CENTER", "0.90"))
DEFAULT_ABETA_SCALE = float(os.environ.get("ACT_ABETA_SCALE", "20.0"))

# Diversity penalty: score *= (1 - γ * max_neighbor_sim). Solves the same
# redundancy problem as GREEDY_COVERAGE; usually one of the two is enough.
DEFAULT_DIVERSITY_GAMMA = float(os.environ.get("ACT_DIVERSITY_GAMMA", "0.0"))

# Region quota: instead of global top-k, distribute slots across regions by
# region saliency, then top-k within each region (with a per-region floor).
DEFAULT_REGION_QUOTA = int(os.environ.get("ACT_REGION_QUOTA", "0"))
DEFAULT_REGION_QUOTA_MIN = int(os.environ.get("ACT_REGION_QUOTA_MIN", "1"))

# Prediction residual: replace saliency with KNN-prediction residual.
DEFAULT_PRED_RESIDUAL = int(os.environ.get("ACT_PRED_RESIDUAL", "0"))
DEFAULT_PRED_KNN = int(os.environ.get("ACT_PRED_KNN", "4"))

# Entropy budget: blend frame-curvature with frame-entropy when allocating
# the per-frame budget. Helpful at very tight budgets and orthogonal to
# coverage-aware greedy.
DEFAULT_ENTROPY_BUDGET = int(os.environ.get("ACT_ENTROPY_BUDGET", "0"))

TAG = "[ACT]"


def _tokens_per_frame(grid_thw_row: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t = int(grid_thw_row[0].item())
    h = int(grid_thw_row[1].item())
    w = int(grid_thw_row[2].item())
    s = max(1, int(spatial_merge_size))
    rh = max(1, h // s)
    rw = max(1, w // s)
    return t, rh, rw, rh * rw


def _region_grid_boundaries(total: int, num_regions: int) -> list[tuple[int, int]]:
    num_regions = max(1, min(num_regions, total))
    base = total // num_regions
    remainder = total % num_regions
    boundaries: list[tuple[int, int]] = []
    start = 0
    for i in range(num_regions):
        end = start + base + (1 if i < remainder else 0)
        boundaries.append((start, end))
        start = end
    return boundaries


def _grid_pool_regions(frames: torch.Tensor, H: int, W: int, num_regions_h: int, num_regions_w: int) -> torch.Tensor:
    T, _, D = frames.shape
    feat_2d = frames.view(T, H, W, D)
    h_bounds = _region_grid_boundaries(H, num_regions_h)
    w_bounds = _region_grid_boundaries(W, num_regions_w)
    region_means = []
    for hs, he in h_bounds:
        for ws, we in w_bounds:
            patch = feat_2d[:, hs:he, ws:we, :]
            region_means.append(patch.reshape(T, -1, D).float().mean(dim=1))
    return torch.stack(region_means, dim=1).to(frames.dtype)


def _build_token_region_map(H: int, W: int, num_regions_h: int, num_regions_w: int, device: torch.device) -> torch.Tensor:
    region_map = torch.zeros(H * W, device=device, dtype=torch.long)
    h_bounds = _region_grid_boundaries(H, num_regions_h)
    w_bounds = _region_grid_boundaries(W, num_regions_w)
    region_idx = 0
    for hs, he in h_bounds:
        for ws, we in w_bounds:
            for hi in range(hs, he):
                for wi in range(ws, we):
                    region_map[hi * W + wi] = region_idx
            region_idx += 1
    return region_map


def _allocate_budget_per_frame(curvature: torch.Tensor, total_budget: int, *, min_k: int, max_k: int) -> torch.Tensor:
    T = int(curvature.shape[0])
    if T <= 0:
        return torch.zeros((0,), device=curvature.device, dtype=torch.long)
    total_budget = int(max(0, min(total_budget, T * max_k)))
    min_k = int(max(0, min(min_k, max_k)))
    min_k_eff = min_k if total_budget >= T * min_k else 0
    base = torch.full((T,), min_k_eff, device=curvature.device, dtype=torch.long)
    remaining = total_budget - int(base.sum().item())
    if remaining <= 0:
        return base
    weights = curvature.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)
    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max(0, max_k - min_k_eff)
    extra = torch.minimum(extra, torch.full_like(extra, max_extra))
    alloc = base + extra
    remaining = total_budget - int(alloc.sum().item())
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if frac[idx].item() < 0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1
    return alloc


def _restore_features(
    kept_feats: torch.Tensor,          # [k, D]
    dropped_feats: torch.Tensor,       # [d, D]
    kept_regions: torch.Tensor,        # [k] long
    dropped_regions: torch.Tensor,     # [d] long
    alpha: float,
    same_region: bool,
) -> torch.Tensor:
    """Winner-takes-all projection of dropped features onto kept features.

    For each dropped token, find the most similar kept token (optionally
    restricted to the same region), then add ``alpha * sim * dropped_feat``
    to that kept token.
    """
    if dropped_feats.numel() == 0 or kept_feats.numel() == 0:
        return kept_feats

    kept_n = F.normalize(kept_feats.float(), dim=-1, eps=1e-6)
    dropped_n = F.normalize(dropped_feats.float(), dim=-1, eps=1e-6)
    sim = dropped_n @ kept_n.t()
    if same_region:
        mask_r = (dropped_regions.unsqueeze(1) == kept_regions.unsqueeze(0)).float()
        sim = sim * mask_r + (mask_r - 1.0) * 1e6

    best_kept = sim.argmax(dim=-1)
    best_sim = sim.max(dim=-1).values.clamp_min(0.0)
    weights = (alpha * best_sim).to(dropped_feats.dtype)
    weighted_drops = dropped_feats * weights.unsqueeze(-1)

    enhanced = kept_feats.clone()
    enhanced.index_add_(0, best_kept, weighted_drops)
    return enhanced


def _compute_pred_residual(ft: torch.Tensor, knn: int = 4) -> torch.Tensor:
    """Spatial prediction residual: how much a token differs from its KNN prediction.

    ft: [N, D]
    returns: [N] float, higher = more unpredictable = more informative.
    """
    N = ft.shape[0]
    if N <= 1:
        return torch.ones(N, device=ft.device, dtype=torch.float32)
    ft_f = ft.float()
    ft_n = F.normalize(ft_f, dim=-1, eps=1e-6)
    sim = ft_n @ ft_n.t()                              # [N, N]
    sim.fill_diagonal_(-float('inf'))
    knn_eff = min(knn, N - 1)
    topk_sim, topk_nn = sim.topk(knn_eff, dim=1)      # [N, knn]
    weights = torch.softmax(topk_sim, dim=1)            # [N, knn]
    predicted = (ft_f[topk_nn] * weights.unsqueeze(-1)).sum(dim=1)  # [N, D]
    residual = (1.0 - F.cosine_similarity(ft_f, predicted, dim=-1, eps=1e-6)).float()
    return residual


def _compute_frame_entropy(frames: torch.Tensor) -> torch.Tensor:
    """Intra-frame token diversity as a proxy for frame complexity.

    frames: [T, N, D]
    returns: [T] float, higher = more diverse tokens = more complex frame.
    """
    frames_n = F.normalize(frames.float(), dim=-1, eps=1e-6)
    frame_means = frames_n.mean(dim=1, keepdim=True)  # [T, 1, D]
    avg_dist = (1.0 - (frames_n * frame_means).sum(dim=-1)).mean(dim=1)  # [T]
    return avg_dist.clamp_min(0.0)


def _compute_avg_sim(frames: torch.Tensor) -> float:
    """Consecutive-frame cosine similarity (mean over tokens per frame).

    frames: [T, N, D]
    returns: scalar in [0, 1].  ~1 = dense sampling, ~0.8 = sparse.
    """
    T = frames.shape[0]
    if T <= 1:
        return 1.0
    frame_means = F.normalize(frames.float().mean(dim=1), dim=-1, eps=1e-6)
    return float((frame_means[:-1] * frame_means[1:]).sum(dim=-1).mean().item())


@torch.no_grad()
def _act_compress_frames(
    frames: torch.Tensor,
    deepstack_frames: List[torch.Tensor],
    H: int, W: int,
    retain_ratio: float,
    min_k: int,
    tag: str = "",
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    T, tokens_per_frame, dim = frames.shape
    if T <= 0 or tokens_per_frame <= 0:
        ki = torch.zeros((0,), device=frames.device, dtype=torch.long)
        return (
            frames.new_empty((0, dim)),
            [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames],
            ki,
        )

    if retain_ratio <= 0:
        retain_ratio = DEFAULT_RETAIN_RATIO

    num_rh = min(DEFAULT_NUM_REGIONS_H, H)
    num_rw = min(DEFAULT_NUM_REGIONS_W, W)
    alpha = float(DEFAULT_PROJECT_ALPHA)
    same_region = bool(DEFAULT_SAME_REGION)

    # --- Direction 1: Adaptive β ---
    if DEFAULT_ADAPTIVE_BETA:
        avg_sim = _compute_avg_sim(frames)
        _s = 1.0 / (1.0 + math.exp(-(avg_sim - DEFAULT_ABETA_CENTER) * DEFAULT_ABETA_SCALE))
        beta = DEFAULT_CURV_WEIGHT_BETA * _s
    else:
        avg_sim = -1.0
        beta = DEFAULT_CURV_WEIGHT_BETA

    region_reps = _grid_pool_regions(frames, H, W, num_rh, num_rw)
    region_reps = F.normalize(region_reps, dim=-1, eps=1e-6)
    region_curvature, frame_curvature = compute_mm_ot_region_curvature(
        region_reps,
        epsilon=DEFAULT_SINKHORN_EPSILON,
        max_iters=DEFAULT_SINKHORN_ITERS,
    )

    token_region_map = _build_token_region_map(H, W, num_rh, num_rw, frames.device)

    # --- Direction B: Frame entropy budget ---
    if DEFAULT_ENTROPY_BUDGET:
        frame_entropy = _compute_frame_entropy(frames)
        if DEFAULT_ADAPTIVE_BETA:
            _alpha_b = 1.0 / (1.0 + math.exp(-(avg_sim - DEFAULT_ABETA_CENTER) * DEFAULT_ABETA_SCALE))
        else:
            _alpha_b = 0.5
        # Normalize both signals to [0, 1] before mixing to avoid scale mismatch
        fc = frame_curvature.float()
        fc_range = fc.max() - fc.min()
        fc_norm = (fc - fc.min()) / (fc_range + 1e-8) if fc_range > 1e-8 else torch.ones_like(fc)
        fe = frame_entropy
        fe_range = fe.max() - fe.min()
        fe_norm = (fe - fe.min()) / (fe_range + 1e-8) if fe_range > 1e-8 else torch.ones_like(fe)
        budget_signal = _alpha_b * fc_norm + (1 - _alpha_b) * fe_norm
        weights = torch.softmax(budget_signal / float(DEFAULT_BUDGET_TEMP), dim=0)
    else:
        weights = torch.softmax(frame_curvature.float() / float(DEFAULT_BUDGET_TEMP), dim=0)

    total_budget = int(round(float(T * tokens_per_frame) * float(retain_ratio)))
    total_budget = max(1, min(T * tokens_per_frame, total_budget))
    min_k_c = int(max(0, min(min_k, tokens_per_frame)))
    if min_k_c > 0:
        total_budget = max(total_budget, T * min_k_c)

    k_t = _allocate_budget_per_frame(weights, total_budget, min_k=min_k_c, max_k=tokens_per_frame)

    actual = int(k_t.sum().item())
    K = region_reps.shape[1]
    _abeta_str = f"avg_sim={avg_sim:.3f} " if DEFAULT_ADAPTIVE_BETA else ""
    print(
        f"{TAG} {tag} T={T} HxW={H}x{W} K={K} beta={beta:.3f} {_abeta_str}"
        f"gamma={DEFAULT_DIVERSITY_GAMMA} rq={DEFAULT_REGION_QUOTA} "
        f"pred_res={DEFAULT_PRED_RESIDUAL} entropy_bgt={DEFAULT_ENTROPY_BUDGET} "
        f"greedy_cov={DEFAULT_GREEDY_COVERAGE} "
        f"alpha={alpha} same_region={same_region} "
        f"eps={DEFAULT_SINKHORN_EPSILON} iters={DEFAULT_SINKHORN_ITERS} "
        f"budget={actual}/{T * tokens_per_frame} ratio={actual / (T * tokens_per_frame):.2%}"
    )

    frame_reps = F.normalize(frames.mean(dim=1), dim=-1, eps=1e-6)

    tokens_out: list[torch.Tensor] = []
    keep_indices: list[torch.Tensor] = []
    deepstack_out: list[list[torch.Tensor]] = [[] for _ in deepstack_frames]

    for fi in range(T):
        k = int(k_t[fi].item())
        if k <= 0:
            continue

        ft = frames[fi]
        rep = frame_reps[fi]

        # --- Direction A: Prediction residual vs outlier ---
        norms = ft.float().norm(dim=-1)
        norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-6)
        if DEFAULT_PRED_RESIDUAL:
            pred_res = _compute_pred_residual(ft, knn=DEFAULT_PRED_KNN)
            saliency = pred_res + norms
        else:
            sim = F.cosine_similarity(ft, rep.unsqueeze(0), dim=-1, eps=1e-6)
            outlier = (1.0 - sim).float()
            saliency = outlier + norms

        rc_t = region_curvature[fi].float()
        rc_min = rc_t.min()
        rc_range = rc_t.max() - rc_min + 1e-6
        rc_norm = (rc_t - rc_min) / rc_range
        token_curv_weight = rc_norm[token_region_map]
        score = saliency * (1.0 + beta * token_curv_weight)

        # --- Direction 2: Diversity penalty ---
        if DEFAULT_DIVERSITY_GAMMA > 0:
            ft_n = F.normalize(ft.float(), dim=-1, eps=1e-6)
            sim_mat = ft_n @ ft_n.t()                              # [N, N]
            sim_mat.fill_diagonal_(-1.0)
            max_neighbor_sim = sim_mat.max(dim=1).values.clamp(0, 1)
            score = score * (1.0 - DEFAULT_DIVERSITY_GAMMA * max_neighbor_sim)

        # --- Direction C: Restore-aware greedy coverage ---
        if DEFAULT_GREEDY_COVERAGE and k < tokens_per_frame:
            pre_k = min(int(k * DEFAULT_GREEDY_PRE_RATIO), tokens_per_frame)
            pre_k = max(pre_k, k)  # ensure at least k candidates
            cand_idx = score.topk(pre_k).indices  # [pre_k]

            cand_n = F.normalize(ft[cand_idx].float(), dim=-1, eps=1e-6)
            all_n = F.normalize(ft.float(), dim=-1, eps=1e-6)
            S = (cand_n @ all_n.t()).clamp(0, 1)  # [pre_k, N]

            coverage = torch.zeros(tokens_per_frame, device=ft.device)
            sel_mask = torch.zeros(pre_k, device=ft.device, dtype=torch.bool)
            sal_w = saliency.unsqueeze(0)  # [1, N]

            for _step in range(k):
                delta = (S - coverage.unsqueeze(0)).clamp_(min=0)  # [pre_k, N]
                gain = (delta * sal_w).sum(dim=1)                  # [pre_k]
                gain.masked_fill_(sel_mask, -1.0)
                best = gain.argmax()                               # stays on GPU
                sel_mask[best] = True
                coverage = torch.max(coverage, S[best])

            sel_indices = sel_mask.nonzero(as_tuple=True)[0]
            # Ensure exactly k tokens (if pre_k was barely enough, pad with top-score unused)
            if sel_indices.numel() < k:
                missing = k - sel_indices.numel()
                unused_scores = score[cand_idx].clone()
                unused_scores[sel_mask] = -float('inf')
                extra = unused_scores.topk(missing).indices
                sel_indices = torch.cat([sel_indices, extra])
            topk_idx = cand_idx[sel_indices[:k]]

        # --- Direction 3: Region quota ---
        elif DEFAULT_REGION_QUOTA:
            K_regions = num_rh * num_rw
            if k >= K_regions:
                region_sal = torch.zeros(K_regions, device=ft.device, dtype=torch.float32)
                for r in range(K_regions):
                    mask_r = (token_region_map == r)
                    if mask_r.any():
                        region_sal[r] = score[mask_r].sum()
                rq_min = min(DEFAULT_REGION_QUOTA_MIN, k // K_regions)
                rq_min = max(rq_min, 1)  # at least 1 per region when budget allows
                region_k = _allocate_budget_per_frame(
                    region_sal, k, min_k=rq_min, max_k=k,
                )
                topk_parts: list[torch.Tensor] = []
                for r in range(K_regions):
                    r_k = int(region_k[r].item())
                    if r_k <= 0:
                        continue
                    mask_r = (token_region_map == r)
                    r_indices = mask_r.nonzero(as_tuple=True)[0]
                    r_k = min(r_k, r_indices.numel())
                    if r_k > 0:
                        r_topk = score[r_indices].topk(r_k).indices
                        topk_parts.append(r_indices[r_topk])
                topk_idx = torch.cat(topk_parts) if topk_parts else torch.zeros(0, device=ft.device, dtype=torch.long)
            else:
                # k < K_regions: not enough budget for per-region allocation, fall back to global top-k
                topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        else:
            topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices

        topk_idx, _ = torch.sort(topk_idx)

        kept_feats = ft.index_select(0, topk_idx)

        if alpha > 0.0 and k < tokens_per_frame:
            drop_mask = torch.ones(tokens_per_frame, dtype=torch.bool, device=ft.device)
            drop_mask[topk_idx] = False
            dropped_idx = drop_mask.nonzero(as_tuple=True)[0]
            dropped_feats = ft.index_select(0, dropped_idx)
            kept_regions = token_region_map[topk_idx]
            dropped_regions = token_region_map[dropped_idx]
            kept_feats = _restore_features(
                kept_feats, dropped_feats,
                kept_regions=kept_regions, dropped_regions=dropped_regions,
                alpha=alpha, same_region=same_region,
            )

            for di, ds in enumerate(deepstack_frames):
                ds_kept = ds[fi].index_select(0, topk_idx)
                ds_dropped = ds[fi].index_select(0, dropped_idx)
                ds_kept = _restore_features(
                    ds_kept, ds_dropped,
                    kept_regions=kept_regions, dropped_regions=dropped_regions,
                    alpha=alpha, same_region=same_region,
                )
                deepstack_out[di].append(ds_kept)
        else:
            for di, ds in enumerate(deepstack_frames):
                deepstack_out[di].append(ds[fi].index_select(0, topk_idx))

        tokens_out.append(kept_feats)
        keep_indices.append(topk_idx + fi * tokens_per_frame)

    if len(tokens_out) == 0:
        ki = torch.zeros((0,), device=frames.device, dtype=torch.long)
        return (
            frames.new_empty((0, dim)),
            [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames],
            ki,
        )

    tokens_comp = torch.cat(tokens_out, dim=0)
    keep_index = torch.cat(keep_indices, dim=0).to(torch.long)
    deepstack_comp = [
        torch.cat(chunks, dim=0) if chunks else ds.new_empty((0, ds.shape[-1]))
        for chunks, ds in zip(deepstack_out, deepstack_frames)
    ]
    return tokens_comp, deepstack_comp, keep_index


def act_compress_qwen3(
    hidden_states: torch.Tensor,
    deepstack_feature_lists: List[torch.Tensor],
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
    retain_ratio: float,
    min_k: int,
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    total_tokens = int(hidden_states.shape[0])
    if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[1] < 3:
        print(f"{TAG} grid_thw invalid; skipping compression.")
        ki = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long)
        return hidden_states, list(deepstack_feature_lists), ki

    tokens_per_frame_list: list[tuple[int, int, int, int]] = []
    expected_total = 0
    for i in range(grid_thw.shape[0]):
        t, rh, rw, tpf = _tokens_per_frame(grid_thw[i], spatial_merge_size)
        tokens_per_frame_list.append((t, rh, rw, tpf))
        expected_total += t * tpf

    if expected_total != total_tokens:
        print(f"{TAG} token count mismatch; expected {expected_total}, got {total_tokens}. Skipping.")
        ki = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long)
        return hidden_states, list(deepstack_feature_lists), ki

    out_tokens: list[torch.Tensor] = []
    out_keep: list[torch.Tensor] = []
    out_deepstack: list[list[torch.Tensor]] = [[] for _ in deepstack_feature_lists]

    offset = 0
    for vid_idx, (t, rh, rw, tpf) in enumerate(tokens_per_frame_list):
        num = t * tpf
        if num <= 0:
            continue
        chunk = hidden_states[offset : offset + num]
        if t <= 1 or tpf <= 0:
            out_tokens.append(chunk)
            out_keep.append(torch.arange(offset, offset + num, device=hidden_states.device, dtype=torch.long))
            for di, ds in enumerate(deepstack_feature_lists):
                out_deepstack[di].append(ds[offset : offset + num])
            offset += num
            continue

        frm = chunk.view(t, tpf, -1)
        ds_frm = [ds[offset : offset + num].view(t, tpf, -1) for ds in deepstack_feature_lists]
        tc, dc, ki = _act_compress_frames(
            frm, ds_frm, H=rh, W=rw,
            retain_ratio=retain_ratio, min_k=min_k,
            tag=f"video[{vid_idx}]",
        )
        out_tokens.append(tc)
        out_keep.append(ki + offset)
        for di, d in enumerate(dc):
            out_deepstack[di].append(d)
        offset += num

    tokens_comp = torch.cat(out_tokens, dim=0) if out_tokens else hidden_states[:0]
    keep_index = (
        torch.cat(out_keep, dim=0).to(torch.long)
        if out_keep
        else hidden_states.new_zeros((0,), dtype=torch.long)
    )
    deepstack_comp = [
        torch.cat(chunks, dim=0) if chunks else ds[:0]
        for chunks, ds in zip(out_deepstack, deepstack_feature_lists)
    ]
    return tokens_comp, deepstack_comp, keep_index


class Qwen3VLVisionModelACT(Qwen3VLVisionModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        compress = bool(kwargs.pop("act_compress", False))
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
        ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs)
            if layer_num in self.deepstack_visual_indexes:
                df = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
                deepstack_feature_lists.append(df)

        hidden_states = self.merger(hidden_states)

        if not compress:
            return hidden_states, deepstack_feature_lists

        is_video = grid_thw is not None and torch.any(grid_thw[:, 0] > 1).item()
        if not is_video:
            ki = torch.arange(int(hidden_states.shape[0]), device=hidden_states.device, dtype=torch.long)
            return hidden_states, deepstack_feature_lists, ki

        tc, dc, ki = act_compress_qwen3(
            hidden_states, deepstack_feature_lists,
            grid_thw=grid_thw,
            spatial_merge_size=int(getattr(self.config, "spatial_merge_size", 1)),
            retain_ratio=DEFAULT_RETAIN_RATIO, min_k=DEFAULT_MIN_K,
        )
        return tc, dc, ki


class Qwen3VLModelACT(Qwen3VLModel):
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        visual_out = self.visual(pixel_values_videos, grid_thw=video_grid_thw, act_compress=True)
        if not isinstance(visual_out, tuple):
            raise ValueError(f"{TAG} Unexpected visual output type: {type(visual_out)}")
        if len(visual_out) == 3:
            return visual_out
        if len(visual_out) == 2:
            ve, dve = visual_out
            ki = torch.arange(int(ve.shape[0]), device=ve.device, dtype=torch.long)
            return ve, dve, ki
        raise ValueError(f"{TAG} Unexpected visual output tuple length: {len(visual_out)}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        is_prefill = False
        if input_ids is not None and input_ids.shape[1] != 1:
            is_prefill = True
        elif cache_position is not None and cache_position[0] == 0:
            is_prefill = True
        elif self.rope_deltas is None:
            is_prefill = True

        if position_ids is None:
            if is_prefill:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids, image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw, attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                if self.rope_deltas is None:
                    self.rope_deltas = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=input_ids.dtype)
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if cache_position is not None and delta.ndim == 2 and delta.shape[0] == 1 and batch_size > 1:
                    delta = delta.expand(batch_size, -1)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        deepstack_video_embeds = None
        if pixel_values_videos is not None and is_prefill:
            if input_ids is not None and input_ids.shape[0] > 1:
                print(f"{TAG} batch>1 detected; disabling compression.")
                video_embeds, deepstack_video_embeds_list = self.visual(
                    pixel_values_videos, grid_thw=video_grid_thw, act_compress=False,
                )
                deepstack_video_embeds = deepstack_video_embeds_list
                _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            else:
                video_embeds, deepstack_video_embeds_list, keep_index_placeholder = self.get_video_features(
                    pixel_values_videos, video_grid_thw,
                )
                deepstack_video_embeds = deepstack_video_embeds_list

                video_token_id = self.config.video_token_id
                video_indices_in_input = (input_ids == video_token_id).nonzero(as_tuple=True)
                tokens_to_keep_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
                total_video_tokens = int(video_indices_in_input[0].shape[0])
                if total_video_tokens > 0:
                    keep_index_placeholder = keep_index_placeholder.to(input_ids.device)
                    safe_idx = keep_index_placeholder.clamp(min=0, max=total_video_tokens - 1)
                    visual_keep_mask = torch.zeros(total_video_tokens, dtype=torch.bool, device=input_ids.device)
                    visual_keep_mask[safe_idx] = True
                    rows_to_drop = video_indices_in_input[0][~visual_keep_mask]
                    cols_to_drop = video_indices_in_input[1][~visual_keep_mask]
                    tokens_to_keep_mask[rows_to_drop, cols_to_drop] = False

                if input_ids.shape[0] == 1:
                    mask_1d = tokens_to_keep_mask[0]
                    input_ids = input_ids[:, mask_1d]
                    inputs_embeds = inputs_embeds[:, mask_1d]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, mask_1d]
                    if position_ids is not None:
                        position_ids = position_ids[:, :, mask_1d]
                        if self.rope_deltas is not None and self.rope_deltas.shape[1] == mask_1d.shape[0]:
                            if self.rope_deltas.dim() == 2:
                                self.rope_deltas = self.rope_deltas[:, mask_1d]
                            elif self.rope_deltas.dim() == 3:
                                self.rope_deltas = self.rope_deltas[:, mask_1d, :]

                _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds_final = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds_final = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                ej = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                ej[image_mask_joint, :] = img_embed
                ej[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds_final.append(ej)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds_final = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds_final = deepstack_video_embeds

        outputs = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds_final, **kwargs,
        )
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
