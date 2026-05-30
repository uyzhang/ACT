<div align="center">

<h1>
  ⚓ ACT: Anchor-Centered Video Token Compression for Efficient Video Large Language Models
</h1>

<p>
  <i>A training-free, plug-and-play framework that organises video token compression around visual anchors.</i>
</p>

</div>

---

## Highlights

ACT compresses video tokens by jointly answering three coupled questions:

1. **(Q1) Anchor budget allocation.** A region-level multi-marginal Sinkhorn estimates cross-frame change (second-order curvature on triplets `(t-1, t, t+1)`) and adaptively distributes the global token budget across frames.
2. **(Q2) Coverage-aware anchor selection.** Within each frame, weighted facility-location greedy fills the anchor slots with representative, non-redundant tokens — replacing pure top-k saliency selection.
3. **(Q3) Residual evidence aggregation.** Non-anchor tokens are pushed forward to their nearest anchor (winner-takes-all, optionally restricted to the same region) and contribute similarity-weighted residuals back into the anchor feature. The token count is unchanged; information density is increased.

ACT is training-free, integrates as a monkey-patch on the Qwen3-VL vision tower, and works on both dense and MoE variants.

---

## Repository layout

```
compressor/act/                       # ACT implementation
├── main.py                           # entry point: act(model)
├── modeling_qwen3_vl_act.py          # Q2 + Q3 + Qwen3-VL hooks
├── curvature.py                      # Q1: batched 3-marginal Sinkhorn
└── README.md

examples/act/                         # quick-start inference scripts
├── inference_qwen3vl_act_32.sh
└── inference_qwen3vl_act_64.sh

lmms_eval/                            # forked from EvolvingLMMs-Lab/lmms-eval
└── models/simple/qwen3_vl.py         # method=act dispatch
```

---

## Setup

```bash
git clone <repo-url> ACT
cd ACT

conda create -n act python=3.10 -y
conda activate act
pip install --upgrade pip
pip install -e ".[train]"
```

ACT requires `transformers` with Qwen3-VL support.

---

## Quick start

Run ACT on Qwen3-VL-8B-Instruct, 64 frames, retain ratio 25%, on VideoMME:

```bash
bash examples/act/inference_qwen3vl_act_64.sh
```

Switch benchmarks:

```bash
TASK=longvideobench bash examples/act/inference_qwen3vl_act_64.sh
TASK=mlvu          bash examples/act/inference_qwen3vl_act_64.sh
TASK=mvbench       bash examples/act/inference_qwen3vl_act_64.sh
```

Switch frame count and retain ratio:

```bash
ACT_RETAIN_RATIO=0.15 bash examples/act/inference_qwen3vl_act_32.sh
```

Ablate the three components (set any of these to `0`):

```bash
ACT_GREEDY_COVERAGE=0 bash examples/act/inference_qwen3vl_act_64.sh   # disable Q2
ACT_PROJECT_ALPHA=0   bash examples/act/inference_qwen3vl_act_64.sh   # disable Q3
ACT_CURV_WEIGHT_BETA=0 bash examples/act/inference_qwen3vl_act_64.sh  # disable Q1 weighting
```

See [`compressor/act/README.md`](compressor/act/README.md) for the full list of environment variables.

---

## Method (in code terms)

| Paper component | Code | Default |
|---|---|---|
| (Q1) Multi-marginal Sinkhorn budget allocation | `compressor/act/curvature.py` | always on |
| (Q2) Coverage-aware greedy selection | `modeling_qwen3_vl_act.py:_act_compress_frames` | `ACT_GREEDY_COVERAGE=1` |
| (Q3) Additive residual pushforward | `modeling_qwen3_vl_act.py:_restore_features` | `ACT_PROJECT_ALPHA=0.3` |

---

## Citation

```bibtex
@article{zhang2026act,
  title={ACT: Anchor-Centered Video Token Compression for Efficient Video Large Language Models},
  author={Zhang, Yi and Liang, Yi-Shan and Yang, Hao-Dong and Guo, Meng-Hao},
  journal={Computational Visual Media},
  year={2026}
}
```

---

## Acknowledgements

This repository builds on top of [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
