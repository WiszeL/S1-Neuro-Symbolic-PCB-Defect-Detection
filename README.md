# PCB Defect Detector And Neurosymbolic Extension

Two layers:

- `neuro/`: frozen DeepPCB detector after Fung et al. (2024)
- `symbolic/` + `neurosym/`: sparse oblique tree student after Hada et al. and Kairgeldin et al.

The Faster R-CNN baseline stays intact as the symbolic stage's frozen teacher.

See `WHAT-I-DID.md` for the full story — problem, build, breakage, fixes, and what the numbers do and don't claim.

Not implemented (on purpose):

- No referential/template-based detector
- No TDD dataset pipeline
- No end-to-end hard-tree training with the detector

## Layout

- `configs/`: model and training settings
- `notebooks/`: orchestration only (01 baseline train → 02 export → 03 tree train → 04 hybrid + heatmaps → 05 Grad-CAM → 06 three-way comparison)
- `neuro/`: Faster R-CNN baseline (records, preprocessing, RCNN input transform)
- `symbolic/`: teacher export, SODT, TAO training, evaluation
- `neurosym/`: hybrid inference + tree-grounded heatmaps
- `gradcam/`: Grad-CAM baseline as the XAI reference point
- `util/`: shared helpers (device/seed/config/io, float32-on-disk features, GT-to-grid projection, shared heatmap metrics, run numbering)
- `dataset/DeepPCB/`: raw data (`trainval.txt` trains, `test.txt` evaluates once)
- `checkpoints/`: auto-numbered `runN.pt` + metrics/history; promoted results copied to `NEWBEST.pt`
- `tests/`: smoke checks (`pytest` or `python tests/test_X.py`)

## Environment

- Python 3.12
- `pip install -r requirements.txt` — `torch`/`torchvision`/`torchaudio` are pinned to `+cu121`; install those three from the CUDA 12.1 index first (or swap wheels for CPU/other CUDA)

## Data

- Raw dataset lives under `dataset/DeepPCB/`, read directly (nested groups, `*_test.jpg` only, `*_temp.jpg` ignored)
- Fixed `640×640` inputs via `dataset.size`
- Train: tensor + `[0,1]` scaling + joint image/box horizontal flip
- Eval: tensor + scaling only, no augmentation
- `RCNNPreprocessing` (a `GeneralizedRCNNTransform` subclass) owns normalization, multi-scale train resizing, fixed eval resizing, batching/padding

## Reproducibility Notes

| Choice | Status |
|---|---|
| Multi-scale sizes, optimizer, warmup, flip, Soft-NMS, RPN IoU rules | From the paper, mirrored in YAML |
| Label ordering, batch size, anchor sizes, pretraining, module internals | Our assumptions, stated in configs/notes |
| ImageNet ResNet-50 + frozen batch norms | Practical small-batch call, not from the paper |
| 15 epochs (paper: 12) | Disclosed, not matched |
| Anchor sizes 32–512 | mmdet defaults; the paper does not state anchors |
| RPN best-anchor rescue only if IoU ≥ 0.3 (`proposal.min_pos_iou`) | From the paper, in `neuro.yaml` |
| Test size 800 (`preprocessing.test_min_size`) | mmdet default and one of the paper's training scales; the paper does not state its test scale |
| VOC all-point AP, legacy +1 box sides (`neuro.train.VOCMeanAP`) | One protocol for every reported number; no COCO scoring |

- **Neck is 64ch, paper says 256.** Cross-stage call, not a shortcut: each tree node fits far fewer weights from the same data at 64ch (D=3136 vs 12544), so fits stay well-posed and splits stay sparse. Cost: absolute AP not comparable to Fung et al. Tables 1/3/4. The 3-way comparison is unaffected (same teacher checkpoint everywhere).
  - Pooled grid is `64×7×7`. SF attention bottleneck `z=32` is our assumption (Fung never sizes it; unrelated to Kairgeldin's `z`).
- **Tree hyperparameters are fixed up front** (`tree_depth`/`l1_lambda`/`sparsity_alpha`/`class_weights` in `configs/symbolic_train.yaml`), like both source papers fix depth before training. One train on `trainval`, one evaluation on `test.txt`. (An 80/20 val sweep existed before; dropped — ~3h per TAO run to pick numbers the papers fix anyway, and it shipped no model.)
- Checkpoints hold `tree_state` (incl. the per-node `margin_exponents` γ of the detection score, fitted after pruning by `symbolic.train.fit_margin_exponents`; `training_config["margin_fit"]` records that fit and `load_neurosymbolic_detector` refuses a tree without it) + `metrics` + `history` + `export_path` + `training_config` (no top-level repeats; `*_metrics.json` repeats them for human readers). `history` has one entry per TAO iteration with a per-node log (`nodes`: status, care rows, solver iterations and stop reason, nonzero weights, seconds), plus a final pruning entry. Only `*_metrics.json` adds `node_sample_counts` / `leaf_class_counts` (rows per node and class mix per leaf, measured on the pruned tree) for the tree diagram.
- Exports use pre-head pooled grids (`box_roi_pool`), never post-MLP embeddings.
- The tree trains offline with TAO on the frozen detector's labels over pre-postprocess RoIs.
- **Node solver.** Each TAO node fit is the same L1-regularized logistic problem the source papers hand to LIBLINEAR (bias penalized, LIBLINEAR's stop rule), solved by `symbolic/tao.py::solve_l1_logistic_reduced_problem` (L-BFGS-B on `w = p - q`, one chunked float32 pass per evaluation) because LIBLINEAR's own memory (~27–32 bytes per matrix element) does not fit a 16 GB machine at this size. `tolerance: 1e-6` solves each node to within float32 precision of its optimum. Validation against LIBLINEAR is in `Contribution.md` Appendix A. sklearn's LIBLINEAR is used only in `tests/`, as the reference.
- **SODT heatmaps** come from tree weights only (no post-hoc saliency), one map per decision node — not one blended map:
  - *leaf-only*: path weights reshaped onto the 7×7 grid. Still in `symbolic/evaluation.py` (training-time ranking, `_row_cell_ranking`), but dropped from reported metrics — superseded everywhere below by the exact map.
  - *exact attribution* (`neurosym/heatmap.py::compute_exact_attribution`): a node's score split onto the pre-pooling FPN map through RoI-Align's own coefficients — sums back exactly, per node or for the whole path (they're additive). See `WHAT-I-DID.md` §8.
  - Read a node's map as "how much this node weighed this region", not "what's here" — region-level (one FPN cell's receptive field is far larger than a mean proposal box), non-negative and normalized per panel (don't compare brightness across panels), and sign-free (a node's LEFT/RIGHT direction lives in its score, not the heatmap color).
- Symbolic vs Grad-CAM metrics: each side masks its own budget fraction at its own resolution (no longer shrunk to a shared coarse grid first) — a fair comparison without handicapping the finer map. Per-node faithfulness (does the region a node weighs decide that node's own routing call?) has no Grad-CAM equivalent — Grad-CAM has no per-step structure to test. See `neurosym/evaluation.py::evaluate_faithfulness_fpn_masking` and both `evaluation.py` docstrings. `use_fpn_heatmap=False` (notebook 06's `USE_FPN_HEATMAP`) scores the displayed 7×7 grid map instead of the exact FPN map, same masking — the resolution ablation.
