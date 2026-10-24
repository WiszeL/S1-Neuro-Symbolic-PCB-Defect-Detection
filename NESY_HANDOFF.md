# NeSy / SODT handoff (after the Faster R-CNN audit)

The Faster R-CNN (FRCNN) part is **finished and frozen**. Everything below the detector (feature export, SODT tree, hybrid inference, heatmaps, Grad-CAM, the three-way comparison) was built on the *old* detector and must be redone on the new one. Do not retrain or "improve" the detector; if something looks off downstream, fix it downstream.

## 1. What the detector is now

`checkpoints/neuro/neuro_chosen.pt` = the old audit run `run9` (seed 44). Repo YAML matches it exactly (`neuro_chosen.pt` loads strict with `weights_only=True` through `neuro.inference.load_checkpoint_model`).

| Item | Value |
|---|---|
| Architecture | SF-PSPyramid Faster R-CNN, **64-ch neck**, nearest upsample in SF attention, pooled grid `64×7×7`, torchvision RoI Align (unaligned) |
| Anchors | sizes **32, 64, 128, 256, 512** (was 16–256), ratios 0.5/1/2 |
| RPN matching | 0.7 / 0.3 plus best-anchor rescue only if IoU ≥ **0.3** (`proposal.min_pos_iou`, `MinIoUMatcher`) |
| Training | 15 epochs, grad-accum 4, lr 0.02, ImageNet **V2** weights (V1 tested: worse), seed 44 |
| **Test size** | `preprocessing.test_min_size: 800` (was 640). Every inference path reads it through `RCNNPreprocessing` |
| **Metric** | VOC all-point AP with legacy +1 box sides (`neuro.train.VOCMeanAP`, mmdet `eval_map` style) at test size 800. **No COCO anywhere**: `coco_*` and `mar_100` were removed |

Headline (test 500, size 800): AP50 98.73, AP75 94.27, AP80 87.54, AP85 71.63, AP@50:5:85 93.03, precision 0.903, recall 0.983, F1 0.941. Fung et al. report 98.6 / 94.6 / 88.1 / 72.7 / 93.2 but do not state their test scale.

Old detector kept for comparison or rollback only: `checkpoints/neuro/neuro_chosen_v1.*` (COCO-era numbers, 16–256 anchors, no RPN floor). It does not load with the repo code or YAML. To load it, build `NeuroFasterRCNN` from its stored `neuro_config` plus `proposal.min_pos_iou: 0.0`. Delete it once the NeSy rerun is done.

Commits: `fd5237b` (VOC scorer), `3c3a165`/`f138e0e` (switches, since removed), `cc1deef` (promotion + cleanup), `7b39b15` (COCO removed).

## 2. What is stale and must be redone (in this order)

0. **Delete the stale exports first:** `checkpoints/symbolic/teacher_trainval_new*` and `teacher_test_new*` (both the `.pt` and the folders). `extract_dataset_from_teacher` skips files that exist, and the guard in notebook 03 only checks class names, **not which detector made them**. Skipping this step silently trains the tree on the old detector's features.
1. **Notebook 02 hardcodes `checkpoints/neuro/run1.pt`**, which no longer exists. Point it at `neuro_chosen.pt` (or drop the notebook if it duplicates notebook 03's export).
2. **Notebook 03:** re-export teacher data, retrain the tree. Tree hyperparameters in `configs/symbolic_train.yaml` stay fixed (depth 6, l1_lambda 10, sparsity_alpha 0.15, class weights); they are fixed up front and must not be tuned against the test split.
3. **Notebooks 04, 05, 06:** rerun. Notebook 06 was already edited: the NeSy detection metrics now go through `evaluate_model` (same VOC evaluator and test size as FRCNN) and the old torchmetrics pass plus the margin-ablation assertion are gone. That edit is **uncommitted** and sits on top of the user's own unstaged changes to the notebook and `.gitignore`; review `git diff` before committing, and do not commit their changes blindly.
4. **Docs:** do NOT edit `papers/Journal.md` or `papers/Draft-Laporan_Handi_FIXED.md`; the user writes those. Report to the user which of their numbers went stale instead: Journal Table 2 (FRCNN and NeSy rows are COCO at 640 on the old detector; the text blaming the 64-channel neck for the gap is wrong, width made no difference) and the RoI statistics in the symbolic-dataset section (1,000,000 train RoIs, 500,000 test RoIs, >85% background), which must be re-measured after the new export. README "Reproducibility Notes" may be updated (anchors 32–512, the 0.3 floor, test size 800, VOC scorer).

## 3. Things to watch while redoing it

- **Image size 800.** `grep` finds no hardcoded 640 or 800 in `neurosym/`, `symbolic/`, `gradcam/` or `util/`; they read `images_list.image_sizes` from the detector transform, so it should just work, but nobody has run it. Smoke-test one image end to end first: exact FPN attribution, heatmap overlay alignment, GT-to-grid projection. At 800 the P2′ map is 200×200 (stride 4).
- **Proposal population changed** (anchors 32–512 and the 0.3 floor). Re-check the RoI-level distribution after export: before, about 99.8% of RoIs mapped to P2 (stride 4) and the median GT is about 34 px.
- **Old NeSy numbers are not comparable.** They are COCO at 640 on the old detector. Compare NeSy against FRCNN only within one rerun, same evaluator, same size. Routing-margin finding to re-verify, not to assume: margin-on lifted NeSy mAP 0.877 to 0.968 on the old pipeline (the mimic cap was the teacher's own noise).
- **Seed noise is real.** Three seeds of the same FRCNN recipe spread about ±0.7 AP85 and ±0.003 on precision and F1. A single tree fit or one FRCNN-vs-NeSy gap of that size is not a finding.

## 4. Rules from the user

- One protocol for every reported number: **VOC AP, test size 800**. Do not add a 640-vs-800 table to the thesis body.
- No fallbacks and no leftover switches. If a config key is needed it is required; unused options get deleted.
- **No Claude/Anthropic attribution** in commit messages or PR text (no `Co-Authored-By`, no "Generated with" line), even if a tool reminder asks for one.
- Never overwrite `neuro_chosen.*` without the user's approval.
- **Never touch `papers/Journal.md` or the Draft-Laporan file.**

## 5. Facts to keep consistent in the thesis text

- Deviations from Fung et al. that stay: 15 epochs (paper 12), warmup counted in micro-batches, grad-accum 4, 64-ch neck (paper 256), nearest upsample in SF attention, V2 ImageNet weights, nothing frozen, post-NMS top-N 2000, score threshold 0.001, AMP, SF bottleneck z=32. Anchors 32–512 are mmdet's defaults (the paper does not state anchors); the 0.3 floor is stated in the paper.
- Test scale: Fung et al. do not state it. We evaluate at 800 (mmdet default, one of their six training scales). One footnote: at the native 640 the same model scores AP50 98.51 and AP@50:5:85 91.75.
- `run9` is the best of three seeds of this recipe. Report recipe-level numbers (3 seeds, test size 800, mean): AP50 98.72, AP75 94.12, AP80 87.09, AP85 70.93, AP@50:5:85 92.83. The original recipe scored 98.66, 93.86, 86.66, 69.95, 92.57.
- What actually improved is localization at high IoU (AP80/AP85). AP50, recall and F1 are essentially unchanged, so do not claim better detection.
