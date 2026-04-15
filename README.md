# PCB Defect Detector And Neurosymbolic Extension

This repository contains two layers of the thesis codebase:

- `neuro/`: the frozen non-referential DeepPCB detector reconstructed from Fung et al. (2024)
- `symbolic/` and `neurosym/`: the teacher-student sparse oblique tree stage grounded in Hada et al. and Kairgeldin et al.

The Faster R-CNN baseline remains intact and is treated as a frozen teacher for the symbolic stage.

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
- `dataset/DeepPCB/`: raw DeepPCB structure with `trainval.txt`, `test.txt`, `groupXXXXX/`, and `*_not/`
- `checkpoints/`: run-based model artifacts and metric/history files
- `tests/`: smoke checks

## Environment Assumptions

- Python 3.11
- Packages are provided by the existing `requirements.txt`
- The code is designed for PyTorch and torchvision from that environment

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

1. `notebooks/01_train_neuro.ipynb`
2. `notebooks/02_inference_and_export_features.ipynb`
3. `notebooks/03_train_symbolic_sodt.ipynb`
4. `notebooks/04_neurosym_inference_heatmap.ipynb`

The notebooks are orchestration-only. Core logic lives under `neuro/`, `symbolic/`, and `neurosym/`.

## Reproducibility Notes

- The paper specifies the multi-scale sizes, optimizer family, warmup, epochs, random flip, Soft-NMS threshold, and RPN anchor IoU rules. Those are reflected directly in the YAML files.
- The paper does not fully specify numeric label ordering, batch size, exact anchor sizes, backbone pretraining, or some module internals. Those are explicit assumptions in the configs and notebook notes.
- The current default assumes ImageNet-pretrained ResNet-50 weights with frozen backbone batch norms, which is a practical small-batch detection assumption rather than an explicitly stated paper detail.
- Each completed training run is saved once as `checkpoints/runN.pt`, `checkpoints/runN_metrics.json`, and `checkpoints/runN_train_history.json`.
- Strict symbolic teacher exports use pooled RoI grids from `box_roi_pool`, not the post-MLP RoI embeddings.
- The symbolic tree is trained offline with TAO-style alternating updates, using the frozen detector's teacher labels on pre-postprocess RoIs.
- Heatmaps in `neurosym/` are derived from the sparse oblique tree path weights reshaped back onto the pooled RoI grid, not from Grad-CAM or another post-hoc saliency method.
