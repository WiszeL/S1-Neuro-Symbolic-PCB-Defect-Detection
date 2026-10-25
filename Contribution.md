# Contribution: replacing the Faster R-CNN classifier with a sparse oblique decision tree (SODT)

Source notes for writing up the Draft Laporan (Bab 3 method, Bab 4 results, limitations). Last updated 2026-10-25. Claude does not edit `papers/Journal.md` or `Draft-Laporan_Handi_FIXED.md`; you write those. Every number below was measured on the test set (500 images, VOC all-point AP, test size 800, IoU 0.5) unless it says trainval.

| Section | Use it for |
|---|---|
| 1 | the claim, in the words to put in the abstract and Bab 1 |
| 2 | Bab 3 (method): detector, tree, the three-part detection score, what was dropped |
| 3 | Bab 4 (results): main table, ablation, cutoff sweep, false positives |
| 4 | what must be disclosed honestly (limitations, Bab 4 and 5) |
| 5 | citations ("dasar") and what not to cite |
| 6, 7 | facts to keep consistent; which draft tables went stale |
| 8 | repository state: code map, checkpoints, rules, what still needs a rerun |
| A | appendix: the LIBLINEAR to L-BFGS-B solver replacement (evidence tables) |

---

## 1. The contribution in a few sentences

The black-box classifier of Faster R-CNN (its softmax head) is replaced **entirely** by a sparse oblique decision tree (SODT, depth 6) that reads the same RoI-pooled features (`64×7×7`). The RPN, the backbone/neck and the box regressor stay; only the classification decision is replaced. The tree gives the class (the label of the leaf a RoI reaches), a detection score read off its own splits, and an explanation (the decision path and per-node maps).

The score uses **no neural classifier output**. It has three parts (§2.3): the **routing margin** gives each detection a confidence, **γ** weights how much each split counts, and the **cutoff** decides which boxes are shown.

Measured result (test 500, VOC, size 800):

| | P | R | F1 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Faster R-CNN (cutoff 0.5) | 0.903 | 0.983 | 0.941 | 0.7964 |
| SODT detector (cutoff 0.7) | 0.913 | 0.981 | 0.946 | 0.7899 |

Precision and F1 are slightly above FRCNN, recall is 0.2 points below, and mAP, which needs no cutoff, is 0.0065 below. **Say "comparable accuracy with an interpretable classifier", not "better detection".** The cutoff 0.7 was chosen with a rule decided after a first test look (§4, item 5).

---

## 2. Method (Bab 3)

### 2.1 The frozen detector

`checkpoints/neuro/neuro_chosen.pt` (audit run `run9`, seed 44). It is finished and frozen; do not retrain it.

| Item | Value |
|---|---|
| Architecture | SF-PSPyramid Faster R-CNN, **64-ch neck**, nearest upsample in SF attention, pooled grid `64×7×7`, torchvision RoI Align (unaligned) |
| Anchors | sizes 32, 64, 128, 256, 512, ratios 0.5 / 1 / 2 |
| RPN matching | 0.7 / 0.3 plus best-anchor rescue only if IoU ≥ 0.3 (`proposal.min_pos_iou`, `MinIoUMatcher`) |
| Training | 15 epochs, grad-accum 4, lr 0.02, ImageNet V2 weights, seed 44 |
| Test size | `preprocessing.test_min_size: 800` |
| Metric | VOC all-point AP, legacy +1 box sides (`neuro.train.VOCMeanAP`, mmdet `eval_map` style). No COCO anywhere |
| Headline | AP50 98.73, AP75 94.27, AP80 87.54, AP85 71.63, AP@50:5:85 93.03, P 0.903, R 0.983, F1 0.941 |

Fung et al. report 98.6 / 94.6 / 88.1 / 72.7 / 93.2 but do not state their test scale.

### 2.2 The SODT classifier head

- **Input and target:** the 3136-dimensional pooled RoI grid; the target is the original classifier's label for each RoI (a mimic / distillation setup). Background RoIs are subsampled to `neg_ratio: 2` per defect RoI at training time.
- **Training:** TAO alternating optimisation of a depth-6 oblique tree with an L1-regularised logistic node solver, then pruning. Settings of the run that is used (`run1`): `l1_lambda 20`, `sparsity_alpha 0.15`, 15 iterations, class weights short 2.0, spur 1.5, open 1.5, pinhole 1.25, spurious_copper 1.25, seed 112. `configs/symbolic_train.yaml` deliberately keeps `l1_lambda 0.01` and `sparsity_alpha 0.85` (not run1's values), so a retrain from the YAML gives a different tree from `run1`; the reported numbers are from `run1_gt_target.pt`, and the cutoffs would need re-deriving for a new tree.
- **Held-out fidelity:** mimic accuracy 0.9627 (run1), macro-F1 against the teacher 0.9079, 42 active internal nodes, 7,327 nonzero weights. Train versus held-out mimic accuracy 0.967 versus 0.963: tree overfit is small.
- **Solver:** LIBLINEAR ran out of memory (27 to 32 bytes per matrix element), so the node solver is replaced by a chunked float32 L-BFGS-B solver of the same L1-logistic problem. This is a contribution to cite; evidence in Appendix A.

### 2.3 The detection score (the three parts)

Pipeline for one RoI: pooled features, tree routing (leaf class + the margins along the path), **score**, Soft-NMS across overlapping boxes, then the **cutoff**. The score is

```
score = Π over active path nodes j of  σ(|w_j·x + b_j|)^γ_j        (γ_j ≥ 0, one γ per internal node)
```

**(1) Routing margin: a confidence.** A leaf only stores a class label, so without a margin every detection scores 1.0. Soft-NMS then has nothing to rank by and keeps an arbitrary box among overlapping ones, and mAP has no ranking either (margin off: mAP 0.64 against 0.79). `σ(|margin|)` of a node says how decisively the RoI fell on its side of that node's split; the product over the path is the detection's confidence. Routing, leaf labels and heatmaps are untouched.

**(2) γ: how much each split counts.** With all γ = 1 every split counts equally. The score is a product of up to six numbers below 1, so a correct detection that went through many splits gets a low score and a fixed cutoff drops it; that was the recall loss (test recall at cutoff 0.5: 0.958 for γ = 1, 0.985 with fitted γ). γ is fitted after TAO and pruning with the tree frozen: log-binomial regression (binomial GLM, log link, no intercept) of `s = exp(Xγ)` on "this RoI is a correct detection", by bounded L-BFGS-B (convex in γ, one optimum). Target = 1 if the ground truth it matched has the leaf's class and IoU ≥ 0.5 (the Faster R-CNN foreground threshold, not tuned), else 0. It is fitted on the 201,191 defect-routed RoIs of the full trainval export (73.6% positives); 14 γ end at exactly 0, meaning that node's doubt is ignored. Side effect: more wrong boxes also score high, which is why precision fell at cutoff 0.5. Fitted γ also ranks better (mAP 0.7818 for γ = 1 against 0.7899).

**(3) Cutoff: which boxes are shown.** A box is shown, and counted for precision and recall, only if its score ≥ cutoff. Boxes below it are dropped. The cutoff is not used in training (the tree learns from the teacher's labels, with no thresholded score) and is not part of Soft-NMS (which has its own floor of 0.001). It is a normal operating point: Faster R-CNN is reported at 0.5 (`precision_score_threshold`, present in the config before this work). The tree's score is on a different scale from a softmax, so NeSy has its own: `nesy_score_threshold: 0.7` (and `nesy_plain_score_threshold: 0.4` for the γ = 1 ablation) in `configs/neuro_train.yaml`. A higher cutoff trades a little recall for precision (test recall 0.985 at 0.5, 0.981 at 0.7). How the 0.7 was chosen: §3.3 and §4.

**Summary:** the margin gives a score, γ makes correct detections score high so recall comes back, and the cutoff removes the extra wrong boxes γ let through.

### 2.4 Explanation

The decision path and the per-node maps (exact FPN attribution, grid 7×7 map) are unchanged by the score. The deletion/insertion AUC calls `tree.route`, so it now uses the γ-weighted confidence: say so in the text, because those AUC values change meaning. Explanation metrics must be rerun on the final tree and cutoff before quoting (§8).

### 2.5 Tried and not part of the method

- **Teacher probability factor.** `score = p_teacher(leaf class) × margin^γ` reached P 0.9229 / R 0.9799 / F1 0.9506 on test, but teacher-only (margin off) already gave F1 0.9450. It keeps the original classifier in the loop, so it contradicts the claim of replacing it. Removed. Do not quote these numbers as NeSy results; if mentioned at all, say once that a classifier-probability factor reached parity but was dropped because it is not a replacement.
- **Per-leaf bias.** `σ(b_leaf + Σγ·log σ)`. On trainval (200 seeded images, cutoff 0.5) P 0.9176 against γ-only 0.9211: no gain. Test was not run; code and checkpoint reverted.

---

## 3. Results (Bab 4)

### 3.1 Main table

Test 500 images, VOC, size 800. P/R at each row's cutoff, IoU 0.5.

| Row | Cutoff | P | R | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0.5 | 0.903 | 0.983 | 0.941 | 0.9873 | 0.7964 |
| **SODT, routing margin + fitted γ** | **0.7** | **0.9129** | **0.9812** | **0.9458** | 0.9817 | 0.7899 |
| SODT, fitted γ (recall-matched cutoff) | 0.6 | 0.8853 | 0.9831 | 0.9316 | 0.9817 | 0.7899 |
| SODT, fitted γ (FRCNN's cutoff) | 0.5 | 0.8632 | 0.9847 | 0.9200 | 0.9817 | 0.7899 |
| Ablation: margin, γ = 1 | 0.4 | 0.9000 | 0.9777 | 0.9373 | 0.9761 | 0.7818 |
| Ablation: margin, γ = 1 | 0.3 | 0.8645 | 0.9834 | 0.9201 | 0.9761 | 0.7818 |
| Ablation: margin off (all scores 1.0) | any | 0.7727 | 0.9873 | 0.8669 | 0.8062 | 0.6396 |

mAP does not depend on the cutoff; that is why the fitted-γ rows share one value. Reading: the margin is what makes a ranking at all (0.64 to 0.79); γ adds a further 0.008 mAP and, at the same cutoff, raises recall; the cutoff moves precision against recall along the same curve.

### 3.2 Why 0.5 was the wrong place to cut

At cutoff 0.5, fitted γ gave P 0.863 against FRCNN's 0.903. A cutoff sweep on 200 seeded trainval images (VOC matcher; P / R) showed the ranking is nearly as good as FRCNN's, only on a different scale:

| Cutoff | Faster R-CNN | SODT fitted γ | SODT γ = 1 |
|---|---|---|---|
| 0.3 | 0.9236 / 0.9993 | 0.8815 / 0.9986 | 0.9199 / 0.9979 |
| 0.4 | 0.9389 / 0.9986 | 0.9035 / 0.9986 | 0.9571 / 0.9936 |
| 0.5 | 0.9458 / 0.9979 | 0.9211 / 0.9979 | 0.9765 / 0.9682 |
| 0.6 | 0.9586 / 0.9972 | 0.9426 / 0.9979 | 0.9885 / 0.8510 |
| 0.7 | 0.9704 / 0.9972 | 0.9579 / 0.9951 | 0.9967 / 0.6314 |

At FRCNN-level recall, fitted γ reaches P 0.943 (cutoff 0.6) and γ = 1 reaches 0.920 (cutoff 0.3): γ improves the ranking. γ = 1 falls off a recall cliff above 0.4.

### 3.3 How the cutoffs were fixed

Both rules were read off the trainval sweep above, none from test.
- **Recall-matched:** the highest cutoff whose trainval recall is at least FRCNN's at 0.5 (fitted γ 0.6, γ = 1 0.3). Decided before the first test look at the new cutoffs.
- **Precision-matched (used in the config):** the lowest cutoff whose trainval precision is at least FRCNN's 0.9458 (fitted γ 0.7, γ = 1 0.4). This rule was added after the 0.6 test row looked low (§4, item 5).

Re-derive both cutoffs if the tree or γ is refit.

### 3.4 False positives (cutoff 0.5, fitted γ, test)

Counted with the P/R matcher (`breakdown_false_positives`): FRCNN 331 (duplicate 3, wrong class 33, background 260); SODT 490 (duplicate 5, wrong class 122, mislocalized 54, background 309; of these, the ones FRCNN shares at the same place: 4 / 31 / 31 / 218). Soft-NMS duplicates are ruled out as the cause. The excess is wrong-class and background boxes. These counts are at the old shared cutoff 0.5; rerun notebook 06 for the 0.7 counts.

Trainval check of where the extra false positives come from (pre-NMS, cutoff 0.5): 9,121 where the tree disagrees with the teacher (8,803 of them where the teacher said background), 54 where the tree agrees with the teacher but the teacher itself was unsure, 17,618 where the teacher is confident too (FRCNN makes the same mistake). So most of the gap is background RoIs the tree sends to a defect leaf.

---

## 4. What to disclose (limitations, Bab 4 and 5)

1. **The first γ fit failed.** It regressed on the teacher's softmax probability, which is high for RoIs the tree misroutes, so γ collapsed toward 0 (root 0.035, 14 of 42 active nodes at exactly 0) and the cutoff filtered almost nothing: P 0.827. Repaired target (ground-truth correctness) gave P 0.863. The repair was picked on trainval evidence, using a decision rule fixed beforehand.
2. **The fit is in-sample.** γ is fitted on the same trainval RoIs the tree trained on; Platt-style fitting would use held-out data. Trainval to test precision drops by about 0.05 (0.9211 to 0.8632 at cutoff 0.5).
3. **The teacher-probability and per-leaf-bias attempts** (§2.5) are dropped; mention the first once, as above.
4. **The score uses no neural classifier output**, but γ was fitted with ground-truth labels on trainval images. That is calibration, not a classifier.
5. **Cutoff selection.** The cutoffs come from trainval, but the sweep was seen before choosing either rule, and the precision-matched rule was added after the recall-matched cutoff (0.6) gave test P 0.885. The test set was therefore evaluated at 0.5, 0.6 and 0.7 for the main row. State that the 0.7 row is a post-hoc operating point, and that mAP is the cutoff-free comparison.
6. **Small sample for the cutoff rules.** On 200 trainval images one box is about 0.07% recall, so 0.6 against 0.7 rests on a few boxes.
7. **Seed noise.** Three seeds of the same FRCNN recipe spread about ±0.7 AP85 and ±0.003 on precision and F1. One tree fit and one FRCNN-vs-SODT gap of that size is not a finding.
8. **Margin-off precision not reconciled.** An earlier remark that margin-off precision beat FRCNN does not match the measured margin-off row (P 0.773, mAP 0.64, with Soft-NMS keeping arbitrary boxes). Not investigated.
9. **Novelty wording.** No published paper fits per-node exponents on a decision tree's path product; this combination is the thesis's adaptation. Unlike a classic log opinion pool, γ has no renormalisation and no sum-to-1 constraint. Unlike temperature scaling, γ can change the ranking of detections, and γ = 0 removes a node's doubt entirely.
10. **Detector provenance.** `run9` is the best of three seeds of its recipe. Report recipe-level numbers (3 seeds, size 800, mean): AP50 98.72, AP75 94.12, AP80 87.09, AP85 70.93, AP@50:5:85 92.83. The original recipe scored 98.66, 93.86, 86.66, 69.95, 92.57. What improved is localisation at high IoU (AP80/AP85); AP50, recall and F1 are essentially unchanged, so do not claim better detection.

---

## 5. Citations ("dasar"), verified 2026-10-25

- Path product as soft-tree path probability: Kontschieder et al. 2015, ICCV, pp. 1467–1475 (already in the draft).
- GLM with log link: McCullagh & Nelder 1989, *Generalized Linear Models*, 2nd ed., Chapman & Hall (book verified; the link-choice section was not checked). R `family` docs list `log` among binomial links; "log link = multiplicative effects" is in arXiv:2301.03349.
- Binomial regression for risk ratios: Wacholder 1986, *Am. J. Epidemiol.* 123(1):174–184 (abstract says "different link functions"; the log link comes from secondary sources).
- Convergence caveat: Williamson, Eliasziw & Fick 2013, *Emerg. Themes Epidemiol.* 10:14. A fitted probability of 1 under the log link is one cause of failed convergence, not the only one.
- Post-hoc fit with the model frozen: Platt (book 2000, preprint 1999; the draft says 1999, pick one), Zadrozny & Elkan 2001 ICML, Niculescu-Mizil & Caruana 2005 ICML. These calibrate scores to true labels, which is what the repaired fit does.
- Exponents on a product of experts, fitted by cross-entropy: Cao & Fleet 2014 (arXiv:1410.7827), Heskes 1998 (NIPS 10, pp. 266–272), Genest & Zidek 1986.
- Distillation: Hinton et al. 2015. It describes the tree's mimic training; it no longer describes the γ fit target (§2.3).
- **Do not** cite McNutt et al. 2003 for the link choice: its abstract never names log-binomial or a log link. Papke & Wooldridge (fractional response) no longer describe the fit target.

---

## 6. Facts to keep consistent in the thesis text

- Deviations from Fung et al. that stay: 15 epochs (paper 12), warmup counted in micro-batches, grad-accum 4, 64-ch neck (paper 256), nearest upsample in SF attention, V2 ImageNet weights, nothing frozen, post-NMS top-N 2000, score threshold 0.001, AMP, SF bottleneck z=32. Anchors 32–512 are mmdet's defaults (the paper does not state anchors); the 0.3 floor is stated in the paper.
- Test scale: Fung et al. do not state it. We evaluate at 800 (mmdet default, one of their six training scales). One footnote: at the native 640 the same model scores AP50 98.51 and AP@50:5:85 91.75.
- One protocol for every reported number: VOC AP, test size 800. Do not add a 640-versus-800 table to the body.
- Old NeSy numbers (COCO at 640 on the old detector) are not comparable; compare only within one rerun, same evaluator, same size. The margin finding "mAP 0.877 to 0.968" is from that old pipeline.
- The cutoff for FRCNN (0.5) and for SODT (0.7) must both be stated wherever P, R or F1 appear.

## 7. Draft items that went stale (you rewrite; Claude does not)

- Journal Table 2: the FRCNN and NeSy rows are COCO at 640 on the old detector. The text blaming the 64-channel neck for the gap is wrong (width made no difference).
- RoI statistics in the symbolic-dataset section (1,000,000 train RoIs, 500,000 test RoIs, more than 85% background): re-measure after the new export.
- Draft Tabel 4.2 and 4.5 to 4.9; Gambar 4.7 and 4.9 to 4.13 (4.13, the per-node panel, now shows γ and σ^γ); the explanation at line ~1890 (FN from background-heavy training does not fit the data); Eq. 3.3 (becomes the γ-weighted product); Tabel 4.10 AUC meaning (§2.4).
- README "Reproducibility Notes" may be updated (anchors 32–512, the 0.3 floor, test size 800, VOC scorer).

---

## 8. Repository state

**Code map.**

| What | Where |
|---|---|
| Tree, `margin_exponents` (required key, no fallback), `path_log_certainties`, confidence `exp(X·γ)` | `symbolic/sodt.py` |
| γ fit: `solve_margin_exponents`, `fit_margin_exponents`, `gt_truth_targets`, `_row_gt_classes` | `symbolic/train.py` |
| Node solver (L-BFGS-B) | `symbolic/tao.py::solve_l1_logistic_reduced_problem` |
| Loader refuses a checkpoint without `training_config["margin_fit"]` | `neurosym/inference.py::load_neurosymbolic_detector` |
| P/R at a chosen cutoff | `neuro/train.py::evaluate_model(..., score_threshold=)` |
| False-positive and missed-detection breakdowns | `neurosym/evaluation.py::breakdown_false_positives`, `breakdown_nesy_misses(..., nesy_score_threshold=)` |
| Cutoffs | `configs/neuro_train.yaml`: `precision_score_threshold 0.5` (FRCNN), `nesy_score_threshold 0.7`, `nesy_plain_score_threshold 0.4` |
| Tests (78 pass) | `tests/test_margin_exponents.py`, `tests/test_miss_breakdown.py`, others |

**Checkpoints** (`checkpoints/symbolic/`).
- `run1_gt_target.pt` (+ `_metrics.json`): **main SODT**; tree-only, γ fitted to the ground-truth target. The cutoffs 0.7 / 0.4 were derived for it.
- `run1.pt`: same tree, γ fitted to the teacher's probability (ablation only).
- `run1_teacher_prob.pt`: **retired, do not load** (its γ was fitted with `log p_teacher` as an offset, so loading it would silently miscalibrate the tree-only score).
- `chosen.pt`: still the old tree; it does not load (no `margin_exponents`, no `margin_fit`). Promoting `run1_gt_target.pt` to `chosen.pt` needs your approval.
- `checkpoints/neuro/neuro_chosen.pt`: frozen detector; never overwrite without approval. `neuro_chosen_v1.*` (old detector) and `symbolic_chosen_d6_l120_a0,15*` can be deleted once notebook 06 is accepted.

**Notebooks.**
- 03 (train, prune, fit γ): fit cell edited, run end to end only through a script on `run1_gt_target.pt`.
- 04 and 06: now read the cutoffs from the config (04 uses 0.7; 06 has a per-row `CUTOFF`: FRCNN 0.5, SODT main 0.7, γ = 1 0.4, margin off 0.7). **Not rerun since**, so their printed numbers are stale.
- 05 (Grad-CAM on FRCNN, cutoff 0.5): no change needed. 02 and 04 load `neuro_chosen.pt` / `chosen.pt`.

**Still to do.** Rerun 04 and 06 (so the tables, FP breakdown and explanation metrics match §3), smoke-test one image end to end at size 800 (exact FPN attribution, overlay alignment, GT-to-grid), decide on promoting `chosen.pt`, then rerun 05.

**Things to watch.** `extract_dataset_from_teacher` skips exports that already exist and only checks class names, not which detector made them; re-export if the detector ever changes. Image size 800 gives a 200×200 P2′ map (stride 4). The RoI distribution changed with the new anchors: re-check it after any re-export.

**Rules from you.** One protocol (VOC, size 800); no fallbacks and no leftover switches (a needed config key is required, unused options are deleted); no Claude/Anthropic attribution in commit messages or PR text; never overwrite `neuro_chosen.*` without approval; never touch `papers/Journal.md` or the Draft-Laporan file; no Python/heredoc replace scripts on your source files (use the editor; notebook 06 is edited as structured JSON because it is too large to open).

---

## Appendix A. SODT node solver: LIBLINEAR replaced (cite in the thesis)

**Why.** sklearn's `liblinear` (L1 logistic) stores every value as a 16-byte node and keeps two copies of the matrix (row and column format), on top of our float32 batch and a float64 upcast. Measured: 27 to 32 bytes per matrix element (3 GB at 36k rows × 3136), against 4 for float32. The root care set of the old tree is 97,923 rows (about 12 GB for LIBLINEAR), the largest 127,038. The new solver needs about the node's float32 batch itself (1.2 to 1.5 GB on 98k to 127k rows; measured peak over baseline within 2% of the batch), so a 16 GB machine trains the tree.

**What.** `symbolic/tao.py::solve_l1_logistic_reduced_problem` solves the same problem as LIBLINEAR's L1R_LR, `min Σ_i (s_i/λ) log(1+exp(-y_i(w·x_i+b))) + |w|_1 + |b|` (bias penalised, as in sklearn's liblinear), with L-BFGS-B on `w = p - q` (p, q ≥ 0), one chunked float32 pass over the node's rows per evaluation, and LIBLINEAR's own stop rule (summed projected-gradient L1 norm ≤ tol · max(min(pos,neg),1)/n · its value at w = 0). TAO's accept-only-if-the-node-score-improves check is unchanged. There is no node-size cap.

**Settings.** `tolerance: 1e-6` (each node solved to within float32 precision of its optimum), `logistic_max_iter: 2000` (slowest fit needed 1006). sklearn's default 1e-4, which the old trees used, stops 0.1 to 4% above the optimum and leaves 6 to 120% extra nonzero weights (node 45: optimum 33 nonzeros, LIBLINEAR at 1e-4 63, ours at 1e-4 73). New trees therefore have fewer nonzero weights than the old ones.

**Evidence** (old tree `chosen.pt` on its stale export, real active-node problems; reference = LIBLINEAR at tol 1e-8, or a float64 run of the same solver where LIBLINEAR cannot fit; gap = (F_ours − F_opt)/F_opt; "agree" = routing-sign agreement with the optimum):

| node | care rows | minority | opt. F | opt. nnz | ours 1e-6: gap / nnz / agree | ours 1e-4: gap / nnz |
|---|---|---|---|---|---|---|
| 45 | 36,500 | 0.455 | 7.55767 | 33 | +1.7e-5 / 33 / 1.0000 | +3.8e-2 / 73 |
| 19 | 21,522 | 0.119 | 8.21317 | 60 | +2.7e-6 / 60 / 1.0000 | +6.5e-3 / 88 |
| 46 | 45,172 | 0.403 | 6.67639 | 43 | +5.1e-6 / 43 / 1.0000 | +1.0e-2 / 67 |
| 39 | 19,311 | 0.018 | 5.06722 | 29 | +6.1e-7 / 29 / 1.0000 | +1.4e-3 / 35 |
| 0 (root, float64 ref.) | 97,923 | 0.217 | 187.39272 | 480 | +4.6e-7 / 482 / 1.0000 | +3.5e-5 / 507 |
| 3 (float64 ref.) | 119,522 | 0.018 | 52.85712 | 245 | +3.0e-6 / 246 / 1.0000 | +5.9e-5 / 256 |
| 7 (float64 ref.) | 117,845 | 0.006 | 27.72475 | 175 | +1.4e-6 / 176 / 1.0000 | +2.4e-5 / 182 |

The float64 reference itself matches LIBLINEAR at tol 1e-8 on node 45 (gap −1.9e-9, same 33 nonzeros). At LIBLINEAR's own production settings (tol 1e-4, 200 iterations) on 14 nodes of 2k to 59k rows, ours at 1e-4 and LIBLINEAR land within about 0.5% of each other in objective (−5.0e-3 to +3.5e-3, routing agreement ≥ 0.9987), but LIBLINEAR itself is 0.1 to 4.4% above the true optimum there. That stop point is path-dependent, so the optimum is the reproducible target, not LIBLINEAR at 1e-4.

**Float32 floor.** Below about 1e-6 the L-BFGS-B line search hits float32 precision and the fit ends `stop_reason: "abnormal"` (it cannot get closer at this precision; not a failure). Most fits of a real run end this way (45 of 63 in the first smoke iteration). The node log records `stop_ratio`: 1 or less means the stop target was reached, R means it stopped at R times the target. A fit at ratio R satisfies LIBLINEAR's own stop rule at tolerance R × 1e-6, so even ratio 186 (1.9e-4) is about sklearn's default 1e-4. Verified against a float64 solve of the same problem:
- cold fits (iteration 1), up to ratio 26.8 on nodes 0, 3, 7: gap ≤ 3e-6, nonzeros within 0.6%; three more nodes at ratio 98.9 (20 rows), 39.5 (6,618 rows) and 29.3 (220 rows): gap ≤ 5e-7, identical nonzeros;
- warm fits (iteration 2), ratio 30 to 319, 119 to 127,966 rows (11 fits): worst gap 3.0e-6, nonzeros differ by at most 4 (of 587), sign agreement 1.0 on all;
- ratios run high because the stop target scales with min(pos, neg)/n, so heavily imbalanced nodes (minority fraction 5e-5 to 8e-4) have targets float32 cannot reach;
- not covered: abnormal fits of iterations 3 and later.

**Cost.** The evaluation is memory-bound, so the solver uses 6 BLAS threads (`threadpoolctl`, only inside the solve) on 1024-row chunks: 62 ms per evaluation on a 120k-row node against 112 ms with all 20 threads. A cold 1e-6 solve on that node takes 64 s (was 114 s) with the same quality. Expect about 1.4 h for a full run if it converges near iteration 8, about 2.6 h if all 15 iterations run (the re-export is extra); keep the machine otherwise idle.

**Smoke run** (new solver on the stale export, 2 TAO iterations; in brackets the same iteration of the old `chosen.pt`: LIBLINEAR, 120k-row subsample cap per node, tol 1e-4):

| | iteration 1 | iteration 2 |
|---|---|---|
| time | 8m00s | 10m34s |
| objective | 104,899 (104,973) | 48,825 (48,822) |
| train mimic accuracy | 0.9502 (0.9505) | 0.9610 (0.9599) |
| nonzero weights | 7,982 (8,163) | 5,059 (5,028) |
| active nodes | 54 (55) | 44 (46) |

Peak anonymous RAM 2.35 GB, 0 fits ended at the iteration cap. The new solver follows the old TAO trajectory to within 0.1% on objective, with a similar tree. Old checkpoints and summaries lack `nodes`, `node_sample_counts` and `leaf_class_counts`, so the notebook cells that draw the tree diagram and dashboard fail with a KeyError on them; retrain first.
