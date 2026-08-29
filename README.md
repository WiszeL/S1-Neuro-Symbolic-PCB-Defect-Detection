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
- `symbolic/`: strict teacher-student export, symbolic dataset handling (including the train/val image split used for hyperparameter selection, see Reproducibility Notes), sparse oblique tree, and TAO training
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
- **SF-PSPyramid output width is 64 channels; Fung et al. specify 256.** This is a deliberate cross-stage design choice, not a memory shortcut: TAO's per-node reduced problem is an L1-logistic fit capped at 30,000 samples (`symbolic/tao.py`), and at 256ch (D=12544) that puts the samples-per-feature ratio at ≈2.4 — well below the region where a regularized logistic fit is well-posed. At 64ch (D=3136) the ratio is ≈9.6. Narrowing the neck keeps TAO's node fits well-posed and the resulting splits sparse and interpretable, which is the actual objective of the symbolic stage. The consequence: the neck architecture (SF-PSPyramid topology, no-lateral-connection rule, L1 regression loss on both heads, Soft-NMS, multi-scale training) is reproduced faithfully, but absolute AP numbers are **not** directly comparable to Fung et al.'s Table 1/3/4 — those were measured at 256ch. The internal three-way comparison (Faster R-CNN vs NeSy vs Grad-CAM in `notebooks/06`) is unaffected, since all three legs run on the same 64ch teacher checkpoint.
- `configs/neuro_train.yaml` trains for 15 epochs; the paper specifies 12. Given the neck-width deviation above already scopes the AP comparison to architecture-only, this is disclosed rather than matched exactly.
- Each completed training run is saved once as `checkpoints/<split>/runN.pt` plus `runN_metrics.json`/`runN_train_history.json` (`util/artifacts.py` picks the next `N`). Reference checkpoints are then copied to a stable name (`NEWBEST.pt`, both `neuro/` and `symbolic/`) so notebooks can point at a fixed path across reruns.
- **SODT hyperparameter selection uses a held-out validation split**, not `test.txt`. `symbolic/dataset.py::_image_level_split_row_indices` deterministically partitions the `trainval` export's *images* (not RoIs) into `train`/`val` (default 80/20, seeded by `split_seed`), governed by the `data.split`/`data.val_fraction`/`data.split_seed` keys in `configs/symbolic_train.yaml`. `notebooks/03_train_symbolic_sodt.ipynb`'s "Hyperparameter Selection" section sweeps `tree_depth`/`l1_lambda`/`sparsity_alpha`/`class_weights` on `train`→`val`, following Hada §4 step 1 ("pick a tree with close to highest validation accuracy and as sparse as possible") and Kairgeldin's (λ, α) regularization-path selection. The selected config is then trained once on the full `trainval` dump and evaluated exactly once on `test.txt`. Note the frozen Faster R-CNN teacher itself trains on all of `trainval` (its own hyperparameters follow Fung et al. directly, with the two disclosed deviations above), so `val` is a model-selection split for the student, not an independent generalization estimate — `test.txt` remains the only split neither stage has seen.
- The SODT checkpoint (`tree_state` + `metrics` + `history` + `export_path` + `training_config`) intentionally does not repeat `class_names`/`feature_shape`/`tree_depth`/`l1_lambda`/`sparsity_alpha` at the top level — those live once in `tree_state`/`training_config`. The paired `*_metrics.json` keeps them at the top level too, since it's meant to be read without loading the tensor payload.
- Strict symbolic teacher exports use pooled RoI grids from `box_roi_pool`, not the post-MLP RoI embeddings.
- The symbolic tree is trained offline with TAO-style alternating updates, using the frozen detector's teacher labels on pre-postprocess RoIs.
- Heatmaps in `neurosym/` come in two forms, both grounded in the sparse oblique tree's own parameters rather than Grad-CAM or another post-hoc saliency method: the *leaf-only* map reshapes the path weights back onto the pooled 7×7 RoI grid; the *exact path attribution* (`neurosym/heatmap.py::compute_exact_attribution`) decomposes the path's score onto the pre-pooling FPN feature map via RoI-Align's own linear coefficients, with no approximation (the per-pixel contribution sums back to the score exactly). See `WHAT-I-DID.md` §8.
- The symbolic and Grad-CAM faithfulness/spatial metrics (`symbolic/evaluation.py`, `gradcam/evaluation.py`) follow a shared perturbation protocol — one 7×7 grid cell (all channels) per unit, ranked by each method's own heatmap, identical cell budgets, each side probing its own model — so the two are directly comparable. See the module docstrings in those files for the full rationale.
