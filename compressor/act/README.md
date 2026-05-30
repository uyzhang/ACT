# ACT — Anchor-Centered video Token Compression

Training-free, plug-and-play video token compression for VideoLLMs that
organises three coupled decisions around visual anchors.

| Paper component                              | Code                          |
|----------------------------------------------|-------------------------------|
| (Q1) Anchor budget allocation (MM-Sinkhorn)  | `curvature.py`                |
| (Q2) Coverage-aware anchor selection (greedy facility location) | `modeling_qwen3_vl_act.py` |
| (Q3) Residual evidence aggregation (additive pushforward) | `modeling_qwen3_vl_act.py` |

## Pipeline

1. The vision tower produces dense per-frame token features.
2. **Q1** — region representations are pooled on a 4×4 grid; a batched
   3-marginal Sinkhorn over consecutive triplets `(t-1, t, t+1)` estimates
   second-order curvature, which is turned into a per-frame anchor budget
   `k_t` summing to `T * tokens_per_frame * retain_ratio`.
3. **Q2** — within each frame we run a saliency-gated facility-location
   greedy on a candidate pool of size `k_t * GREEDY_PRE_RATIO`, picking
   anchors that maximise marginal coverage of the dense token field.
4. **Q3** — every non-anchor token is assigned (winner-takes-all, optionally
   restricted to the same region) to its nearest anchor, contributing a
   similarity-weighted residual to that anchor's feature. Token count is
   unchanged; the same pushforward is applied to every deepstack scale.

## Environment variables

| Variable                  | Default | Notes                                        |
|---------------------------|---------|----------------------------------------------|
| `ACT_RETAIN_RATIO`        | 0.25    | Global keep ratio                             |
| `ACT_MIN_K`               | 1       | Per-frame minimum anchor count                |
| `ACT_BUDGET_TEMP`         | 0.7     | Softmax temperature for budget allocation     |
| `ACT_NUM_REGIONS_H/W`     | 4 / 4   | Region grid                                   |
| `ACT_CURV_WEIGHT_BETA`    | 1.0     | Region-curvature weight in saliency           |
| `ACT_SINKHORN_EPSILON`    | 0.05    | MM-OT entropy regularisation                  |
| `ACT_SINKHORN_ITERS`      | 50      | MM-OT iterations                              |
| `ACT_GREEDY_COVERAGE`     | 1       | Coverage-aware anchor selection (Q2)          |
| `ACT_GREEDY_PRE_RATIO`    | 2.0     | Candidate pool size multiplier                |
| `ACT_PROJECT_ALPHA`       | 0.3     | Residual aggregation strength α (Q3)          |
| `ACT_SAME_REGION`         | 1       | Restrict pushforward to the same region       |

## Usage

```bash
METHOD=act bash examples/act/inference_qwen3vl_act_64.sh
```
