"""Multi-marginal Sinkhorn solver for second-order region curvature.

Solves a 3-way entropy-regularised optimal transport problem over triplets of
consecutive frames.  The cost function directly encodes direction change
(second-order curvature), so the matching and curvature computation are unified
into a single optimisation.
"""

import torch
import torch.nn.functional as F


def multi_marginal_sinkhorn(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iters: int = 50,
) -> torch.Tensor:
    """Solve 3-marginal entropy-regularised OT with uniform marginals.

    Parameters
    ----------
    cost : Tensor [K, K, K]
        Non-negative cost tensor.  Indices correspond to regions in
        frame t-1, frame t, and frame t+1 respectively.
    epsilon : float
        Entropy regularisation strength.
    max_iters : int
        Number of alternating-normalisation iterations.

    Returns
    -------
    Tensor [K, K, K]
        Triply-stochastic transport tensor where each 2D marginal sums to
        1/K along the remaining axis.
    """
    K = cost.shape[0]
    if K <= 1:
        return torch.ones((K, K, K), device=cost.device, dtype=cost.dtype)

    log_K = torch.tensor(K, device=cost.device, dtype=cost.dtype).log()
    log_Pi = -cost.float() / max(epsilon, 1e-8)

    for _ in range(max_iters):
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(1, 2), keepdim=True) - log_K
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(0, 2), keepdim=True) - log_K
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(0, 1), keepdim=True) - log_K

    return log_Pi.exp().to(cost.dtype)


def _build_direction_change_cost(
    reps_prev: torch.Tensor,
    reps_curr: torch.Tensor,
    reps_next: torch.Tensor,
) -> torch.Tensor:
    """Build the 3D cost tensor C[i, j, k] = 1 - cos(v_in, v_out).

    Parameters
    ----------
    reps_prev, reps_curr, reps_next : Tensor [K, D]
        L2-normalised region representations for three consecutive frames.

    Returns
    -------
    Tensor [K, K, K]  (indices: prev_region, curr_region, next_region)
    """
    reps_prev = reps_prev.float()
    reps_curr = reps_curr.float()
    reps_next = reps_next.float()

    # v_in[i, j, :] = reps_curr[j] - reps_prev[i]   shape [K_i, K_j, D]
    v_in = reps_curr.unsqueeze(0) - reps_prev.unsqueeze(1)
    # v_out[j, k, :] = reps_next[k] - reps_curr[j]   shape [K_j, K_k, D]
    v_out = reps_next.unsqueeze(0) - reps_curr.unsqueeze(1)

    v_in_n = F.normalize(v_in, dim=-1, eps=1e-6)
    v_out_n = F.normalize(v_out, dim=-1, eps=1e-6)

    # cos[i, j, k] = v_in_n[i, j, :] · v_out_n[j, k, :]
    cos_sim = torch.einsum("ijd,jkd->ijk", v_in_n, v_out_n)
    return (1.0 - cos_sim).clamp_min(0.0)


def compute_mm_ot_region_curvature(
    region_reps: torch.Tensor,
    epsilon: float = 0.05,
    max_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-region second-order curvature via multi-marginal OT (batched).

    For each triplet of consecutive frames (t-1, t, t+1), solves a 3-way OT
    problem whose cost directly encodes direction change.  The residual
    transport cost marginalised over the centre frame gives per-region
    curvature.

    This implementation batches all T-2 triplets into a single tensor
    operation for efficient GPU utilization.

    Parameters
    ----------
    region_reps : Tensor [T, K, D]  (L2-normalised)
    epsilon : float
    max_iters : int

    Returns
    -------
    region_curvature : Tensor [T, K]
    frame_curvature  : Tensor [T]
    """
    T, K, _ = region_reps.shape
    device = region_reps.device
    dtype = region_reps.dtype

    region_curvature = torch.zeros((T, K), device=device, dtype=torch.float32)

    if T <= 2:
        region_curvature[:] = 1.0 / K
        return region_curvature.to(dtype), region_curvature.sum(dim=1).to(dtype)

    # Batch all T-2 triplets at once
    r_prev = F.normalize(region_reps[:-2].float(), dim=-1, eps=1e-6)  # [T-2, K, D]
    r_curr = F.normalize(region_reps[1:-1].float(), dim=-1, eps=1e-6)  # [T-2, K, D]
    r_next = F.normalize(region_reps[2:].float(), dim=-1, eps=1e-6)    # [T-2, K, D]

    # Build batched cost tensor [T-2, K, K, K]
    C = _build_direction_change_cost_batched(r_prev, r_curr, r_next)

    # Batched Sinkhorn [T-2, K, K, K]
    Pi_star = _multi_marginal_sinkhorn_batched(C, epsilon=epsilon, max_iters=max_iters)

    # Curvature for region j: sum over matched (i, k)
    kappa = (Pi_star * C).sum(dim=(1, 3))  # [T-2, K]
    region_curvature[1:T-1] = kappa

    region_curvature[0] = 1.0 / K
    region_curvature[T - 1] = 1.0 / K

    frame_curvature = region_curvature.sum(dim=1)
    return region_curvature.to(dtype), frame_curvature.to(dtype)


def _build_direction_change_cost_batched(
    reps_prev: torch.Tensor,
    reps_curr: torch.Tensor,
    reps_next: torch.Tensor,
) -> torch.Tensor:
    """Build batched 3D cost tensor C[b, i, j, k] = 1 - cos(v_in, v_out).

    Parameters
    ----------
    reps_prev, reps_curr, reps_next : Tensor [B, K, D]

    Returns
    -------
    Tensor [B, K, K, K]
    """
    # v_in[b, i, j, :] = reps_curr[b, j] - reps_prev[b, i]
    v_in = reps_curr.unsqueeze(1) - reps_prev.unsqueeze(2)  # [B, K, K, D]
    # v_out[b, j, k, :] = reps_next[b, k] - reps_curr[b, j]
    v_out = reps_next.unsqueeze(1) - reps_curr.unsqueeze(2)  # [B, K, K, D]

    v_in_n = F.normalize(v_in, dim=-1, eps=1e-6)
    v_out_n = F.normalize(v_out, dim=-1, eps=1e-6)

    # cos[b, i, j, k] = v_in_n[b, i, j, :] . v_out_n[b, j, k, :]
    cos_sim = torch.einsum("bijd,bjkd->bijk", v_in_n, v_out_n)
    return (1.0 - cos_sim).clamp_min(0.0)


def _multi_marginal_sinkhorn_batched(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iters: int = 50,
) -> torch.Tensor:
    """Batched 3-marginal entropy-regularised OT with uniform marginals.

    Parameters
    ----------
    cost : Tensor [B, K, K, K]
    epsilon : float
    max_iters : int

    Returns
    -------
    Tensor [B, K, K, K]
    """
    K = cost.shape[1]
    if K <= 1:
        return torch.ones_like(cost)

    log_K = torch.tensor(K, device=cost.device, dtype=cost.dtype).log()
    log_Pi = -cost.float() / max(epsilon, 1e-8)  # [B, K, K, K]

    for _ in range(max_iters):
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(2, 3), keepdim=True) - log_K
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(1, 3), keepdim=True) - log_K
        log_Pi = log_Pi - torch.logsumexp(log_Pi, dim=(1, 2), keepdim=True) - log_K

    return log_Pi.exp().to(cost.dtype)
