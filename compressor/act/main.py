"""ACT: Anchor-Centered video Token compression — entry point.

Patches Qwen3-VL (and the MoE variant) so that visual encoding goes through
the ACT pipeline:

    (1) anchor budget allocation via region-level multi-marginal Sinkhorn
    (2) coverage-aware anchor selection via weighted facility-location greedy
    (3) residual evidence aggregation via additive pushforward to nearest anchor
"""

import types

from .modeling_qwen3_vl_act import (
    Qwen3VLModelACT,
    Qwen3VLVisionModelACT,
)


def act(model):
    print("################################")
    print("############# ACT ##############")
    print("################################")
    from .modeling_qwen3_vl_act import (
        DEFAULT_RETAIN_RATIO, DEFAULT_MIN_K, DEFAULT_BUDGET_TEMP,
        DEFAULT_NUM_REGIONS_H, DEFAULT_NUM_REGIONS_W,
        DEFAULT_CURV_WEIGHT_BETA,
        DEFAULT_SINKHORN_EPSILON, DEFAULT_SINKHORN_ITERS,
        DEFAULT_PROJECT_ALPHA, DEFAULT_SAME_REGION,
        DEFAULT_GREEDY_COVERAGE, DEFAULT_GREEDY_PRE_RATIO,
    )
    print(
        f"[ACT] retain_ratio={DEFAULT_RETAIN_RATIO} min_k={DEFAULT_MIN_K} temp={DEFAULT_BUDGET_TEMP} "
        f"regions={DEFAULT_NUM_REGIONS_H}x{DEFAULT_NUM_REGIONS_W} beta={DEFAULT_CURV_WEIGHT_BETA} "
        f"sinkhorn_eps={DEFAULT_SINKHORN_EPSILON} sinkhorn_iters={DEFAULT_SINKHORN_ITERS} "
        f"project_alpha={DEFAULT_PROJECT_ALPHA} same_region={DEFAULT_SAME_REGION} "
        f"greedy_coverage={DEFAULT_GREEDY_COVERAGE} greedy_pre_ratio={DEFAULT_GREEDY_PRE_RATIO}"
    )

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLVisionModel,
    )

    Qwen3VLVisionModel.forward = Qwen3VLVisionModelACT.forward
    Qwen3VLModel.get_video_features = Qwen3VLModelACT.get_video_features
    Qwen3VLModel.forward = Qwen3VLModelACT.forward

    patched_moe_visual = []
    try:
        import importlib
        moe_modeling = importlib.import_module(
            "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"
        )
    except Exception:
        moe_modeling = None

    if moe_modeling is not None:
        for _, obj in vars(moe_modeling).items():
            if not isinstance(obj, type):
                continue
            cls_name = getattr(obj, "__name__", str(obj))
            looks_like_vision_backbone = (
                ("VisionModel" in cls_name and hasattr(obj, "forward"))
                or (
                    hasattr(obj, "forward")
                    and hasattr(obj, "patch_embed")
                    and hasattr(obj, "blocks")
                )
            )
            if not looks_like_vision_backbone:
                continue
            setattr(obj, "forward", Qwen3VLVisionModelACT.forward)
            patched_moe_visual.append(cls_name)

    if patched_moe_visual:
        names = ", ".join(sorted(set(patched_moe_visual)))
        print(f"[ACT] Patched MoE vision classes: {names}")

    try:
        backbone = getattr(model, "model", None)
        visual = getattr(backbone, "visual", None) if backbone is not None else None
        if visual is not None and callable(getattr(visual, "forward", None)):
            visual.forward = types.MethodType(
                Qwen3VLVisionModelACT.forward, visual
            )
            print(f"[ACT] Patched runtime visual instance: {type(visual).__name__}")
    except Exception as exc:
        print(f"[ACT] Warning: failed to patch runtime visual instance: {exc}")

    return model
