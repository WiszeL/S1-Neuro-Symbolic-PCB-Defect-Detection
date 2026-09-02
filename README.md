# PCB Defect Detector And Neurosymbolic Extension

This repository contains two layers of the thesis codebase:

- `neuro/`: the frozen non-referential DeepPCB detector reconstructed from Fung et al. (2024)
- `symbolic/` and `neurosym/`: the teacher-student sparse oblique tree stage grounded in Hada et al. and Kairgeldin et al.

The Faster R-CNN baseline remains intact and is treated as a frozen teacher for the symbolic stage.

See `WHAT-I-DID.md` for the chronological narrative of this work — the problem it addresses,
what was built, what broke, how it was diagnosed and fixed, and what the final results do and
do not claim.

What is intentionally not implemented:
- No referential/template-based detector
- No TDD dataset pipeline
- No end-to-end hard-tree training with the detector

## Project Layout

- `configs/`: explicit model and training settings
- `notebooks/`: orchestration notebooks for baseline training, baseline inference, symbolic training, and hybrid inference
- `neuro/`: pure Faster R-CNN baseline code
  - `prepare_dataset.py`: DeepPCB dataset records, image loading, and torchvision `BoundingBoxes` target preparation
  - `preprocess_dataset.py`: torchvision-native train/eval preprocessing and Faster R-CNN input preprocessing
- `symbolic/`: strict teacher-student export, symbolic dataset handling, sparse oblique tree, and TAO training
- `neurosym/`: frozen-detector + symbolic-classifier hybrid inference and symbolic heatmaps
- `gradcam/`: Grad-CAM baseline (backbone hook, evaluation metrics, comparison plots) used as an XAI reference point against the symbolic heatmaps
- `util/`: shared helpers used across `neuro`/`symbolic`/`neurosym`/`gradcam` — `device.py`/`seed.py`/`config.py`/`io.py`/`features.py`, `geometry.py` (GT-to-RoI-grid projection), `heatmap_metrics.py` (normalize/pointing-score/top-k overlap/importance-ranking/random-baseline/stratified spatial results — shared by the symbolic and Grad-CAM evaluators so both sides use identical logic), `visualization.py` (tensor-to-image), `artifacts.py` (run numbering)
- `dataset/DeepPCB/`: raw DeepPCB structure with `trainval.txt`, `test.txt`, `groupXXXXX/`, and `*_not/`
- `checkpoints/`: model artifacts and metric/history files. Auto-numbered runs use `runN.pt`/`runN_metrics.json`; checkpoints promoted as the reference result are copied to `NEWBEST.pt` (both `neuro/` and `symbolic/`)
- `tests/`: smoke checks (plain `assert`-based, runnable via `pytest` or directly with `python tests/test_X.py`)

## Environment Assumptions

- Python 3.12
- `pip install -r requirements.txt`. `torch`/`torchvision`/`torchaudio` are pinned to the `+cu121` CUDA 12.1 build; install those three first from the PyTorch CUDA 12.1 index (or swap in CPU/other-CUDA wheels) before installing the rest, or drop the `+cu121` suffix for a CPU environment.

## DeepPCB Setup

- Keep the raw dataset under `dataset/DeepPCB/`
- Use the provided `trainval.txt` for training and `test.txt` only for the final evaluation
- The loader reads the raw nested groups directly
- Non-referential mode uses `*_test.jpg` and ignores `*_temp.jpg`
- DeepPCB images are treated as fixed-size `640 x 640` inputs through `dataset.size` in `configs/neuro_train.yaml`

## Dataset And Preprocessing

- `prepare_dataset.py` prepares dataset records and targets. It keeps image loading in `PCBDataset.__getitem__()` and stores annotations as torchvision `tv_tensors.BoundingBoxes` with the configured DeepPCB canvas size.
- `preprocess_dataset.py` owns image preprocessing. Training uses `ToImage`, `ToDtype(torch.float32, scale=True)`, and torchvision v2 `RandomHorizontalFlip`, so image and bounding boxes are transformed together.
- Evaluation preprocessing only converts images to tensors and scales pixel values to `[0, 1]`; it does not apply random augmentation.
- Faster R-CNN input preprocessing is handled by `RCNNPreprocessing`, a `GeneralizedRCNNTransform` subclass. It applies ImageNet mean/std normalization, multi-scale resizing during training, fixed-size resizing during evaluation, and internal batching/padding at model forward time.

## Running The Workflow

Run the notebooks in order:

1. `notebooks/01_train_neuro.ipynb` — train the Faster R-CNN baseline
2. `notebooks/02_inference_and_export_features.ipynb` — export teacher RoI features for the symbolic stage
3. `notebooks/03_train_symbolic_sodt.ipynb` — train and prune the SODT student
4. `notebooks/04_neurosym_inference_heatmap.ipynb` — hybrid neuro-symbolic inference with heatmap explanations
5. `notebooks/05_gradcam_baseline.ipynb` — Grad-CAM baseline explanations for comparison
6. `notebooks/06_three_way_comparison.ipynb` — Faster R-CNN vs neuro-symbolic vs Grad-CAM, detection and explanation metrics side by side

The notebooks are orchestration-only. Core logic lives under `neuro/`, `symbolic/`, `neurosym/`, and `gradcam/`.

## Reproducibility Notes

- The paper specifies the multi-scale sizes, optimizer family, warmup, epochs, random flip, Soft-NMS threshold, and RPN anchor IoU rules. Those are reflected directly in the YAML files.
- The paper does not fully specify numeric label ordering, batch size, exact anchor sizes, backbone pretraining, or some module internals. Those are explicit assumptions in the configs and notebook notes.
- The current default assumes ImageNet-pretrained ResNet-50 weights with frozen backbone batch norms, which is a practical small-batch detection assumption rather than an explicitly stated paper detail.
- **SF-PSPyramid output width is 256 channels, matching Fung et al.** An earlier version of this codebase ran the neck at 128ch to keep the next stage's TAO node fits (an L1-logistic regression per internal node) well-posed on a narrower pooled feature. That choice was retired once `symbolic/tao.py` gained an explicit solver sample cap (see below): the cap addresses well-posedness directly at the node level, so the neck no longer has to diverge from the paper for it. The pooled RoI grid is now `256×7×7`. Fung et al.'s SF attention (§3.3) compresses the pooled descriptor to `z` channels before expanding back; the symbol is theirs but the *value* of `z` is never stated. It is set to 64 here as an assumption (`SFAttention` in `neuro/faster_rcnn.py`). Not to be confused with Kairgeldin's `z`, which is the feature vector fed to the tree.
- **TAO solver sample cap: 120,000 (Hada/Kairgeldin fit each node's full reduced set).** `symbolic/tao.py`'s `solver_cap` limits the LIBLINEAR fit at any internal node whose reduced set exceeds it. LIBLINEAR is double-precision and materializes the feature matrix densely, so at 256ch (D=12544) an uncapped root reduced set (~350k rows on the `trainval` export) needs roughly 50 GB of RAM. Only the root and the top two or three nodes ever exceed 120k; deeper nodes fit on their full set. At the cap the samples-per-feature ratio is ≈9.6 (well-posed), the subsample is uniform, and each node's regularization `C` is still computed from the true `|R_i|`, so only the solver sample count changes, not the regularization strength. `0` disables the cap. The effect on the learned splits versus an uncapped fit has not been measured.
- `configs/neuro_train.yaml` trains for 15 epochs; the paper specifies 12. Disclosed rather than matched exactly; a retrain to shave three epochs was not judged worth it. This is the only teacher-side deviation from Fung et al.
- Each completed training run is saved once as `checkpoints/<split>/runN.pt` plus `runN_metrics.json`/`runN_train_history.json` (`util/artifacts.py` picks the next `N`). Reference checkpoints are then copied to a stable name (`NEWBEST.pt`, both `neuro/` and `symbolic/`) so notebooks can point at a fixed path across reruns.
- **SODT hyperparameters are fixed a priori in `configs/symbolic_train.yaml`** (`search.tree_depth`/`l1_lambda`/`sparsity_alpha`/`class_weights`), not chosen by a search against any evaluation split. This follows both source papers' own practice of fixing tree depth before training rather than tuning it (Hada §6.1 depth 6, §6.2 depth 5; Kairgeldin §6 "we restrict tree depth to 5"). The student trains once on the full `trainval` dump and is evaluated exactly once on `test.txt`. An earlier version of this codebase ran a validation sweep (an 80/20 image-level split of `trainval`) to select these values; it was removed because at ~3 h per TAO run it cost three extra full trainings to choose four numbers the source papers fix a priori, and it shipped no model of its own. The frozen Faster R-CNN teacher also trains on all of `trainval`, so `test.txt` remains the only split neither stage has seen.
- The SODT checkpoint (`tree_state` + `metrics` + `history` + `export_path` + `training_config`) intentionally does not repeat `class_names`/`feature_shape`/`tree_depth`/`l1_lambda`/`sparsity_alpha` at the top level — those live once in `tree_state`/`training_config`. The paired `*_metrics.json` keeps them at the top level too, since it's meant to be read without loading the tensor payload.
- Strict symbolic teacher exports use pooled RoI grids from `box_roi_pool`, not the post-MLP RoI embeddings.
- The symbolic tree is trained offline with TAO-style alternating updates, using the frozen detector's teacher labels on pre-postprocess RoIs.
- Heatmaps in `neurosym/` come in two forms, both grounded in the sparse oblique tree's own parameters rather than Grad-CAM or another post-hoc saliency method: the *leaf-only* map reshapes the path weights back onto the pooled 7×7 RoI grid; the *exact path attribution* (`neurosym/heatmap.py::compute_exact_attribution`) decomposes the path's score onto the pre-pooling FPN feature map via RoI-Align's own linear coefficients, with no approximation (the per-pixel contribution sums back to the score exactly). See `WHAT-I-DID.md` §8.
- The symbolic and Grad-CAM faithfulness/spatial metrics (`symbolic/evaluation.py`, `gradcam/evaluation.py`) follow a shared perturbation protocol — one 7×7 grid cell (all channels) per unit, ranked by each method's own heatmap, identical cell budgets, each side probing its own model — so the two are directly comparable. See the module docstrings in those files for the full rationale.
