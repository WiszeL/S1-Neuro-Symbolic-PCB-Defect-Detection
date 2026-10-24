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
1. **Notebook 02 hardcodes `checkpoints/neuro/run1.pt`**, which no longer exists. Point it at `neuro_chosen.pt` (or drop the notebook if it duplicates notebook 03's export). *Done (§8): notebooks 02 and 04 now load `neuro_chosen.pt` / `chosen.pt`.*
2. **Notebook 03:** re-export teacher data, retrain the tree, then fit the margin exponents (new cell, §8). Tree hyperparameters in `configs/symbolic_train.yaml` stay fixed (depth 6, l1_lambda 20, sparsity_alpha 0.15, class weights incl. spurious_copper 1.25, as in `checkpoints/symbolic/chosen.pt`); they are fixed up front and must not be tuned against the test split.
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

## 6. SODT node solver: LIBLINEAR replaced (cite this in the thesis)

**Why.** sklearn's `liblinear` (L1 logistic) stores every value as a 16-byte node and keeps two copies of the matrix (row + column format), on top of our float32 batch and a float64 upcast. Measured: 27 to 32 bytes per matrix element (3 GB at 36k rows x 3136), against 4 for float32. Root care set of the old tree is 97,923 rows (about 12 GB for LIBLINEAR), the largest is 127,038. The new solver needs about the node's float32 batch itself (1.2 to 1.5 GB on 98k to 127k rows, measured peak over baseline within 2% of the batch), against LIBLINEAR's extra 27 to 32 bytes per element (7 to 8x the batch), so a 16 GB machine trains the tree.

**What.** `symbolic/tao.py::solve_l1_logistic_reduced_problem` solves the same problem as LIBLINEAR's L1R_LR, `min sum_i (s_i/lambda) log(1+exp(-y_i(w.x_i+b))) + |w|_1 + |b|` (bias penalized, as in sklearn's liblinear), with L-BFGS-B on `w = p - q` (p, q >= 0), one chunked float32 pass over the node's rows per evaluation, and LIBLINEAR's own stop rule (summed projected-gradient L1 norm <= tol * max(min(pos,neg),1)/n * its value at w = 0). TAO's accept-only-if-the-node-score-improves check is unchanged. There is no node-size cap.

**Settings.** `tolerance: 1e-6` (solve each node to within float32 precision of its optimum), `logistic_max_iter: 2000` (L-BFGS-B iterations; slowest fit needed 1006). sklearn's default 1e-4, which the old trees used, stops 0.1 to 4% above the optimum and leaves 6 to 120% extra nonzero weights (node 45: optimum 33 nonzeros, LIBLINEAR at 1e-4 63, ours at 1e-4 73). Nonzero counts of the new trees are therefore expected to be lower than the old ones.

**Evidence** (old tree `chosen.pt` on its stale export, real active-node problems; reference = LIBLINEAR at tol 1e-8, or a float64 run of the same solver where LIBLINEAR cannot fit; gap = (F_ours - F_opt)/F_opt, "agree" = routing-sign agreement with the optimum):

| node | care rows | minority | opt. F | opt. nnz | ours 1e-6: gap / nnz / agree | ours 1e-4: gap / nnz |
|---|---|---|---|---|---|---|
| 45 | 36,500 | 0.455 | 7.55767 | 33 | +1.7e-5 / 33 / 1.0000 | +3.8e-2 / 73 |
| 19 | 21,522 | 0.119 | 8.21317 | 60 | +2.7e-6 / 60 / 1.0000 | +6.5e-3 / 88 |
| 46 | 45,172 | 0.403 | 6.67639 | 43 | +5.1e-6 / 43 / 1.0000 | +1.0e-2 / 67 |
| 39 | 19,311 | 0.018 | 5.06722 | 29 | +6.1e-7 / 29 / 1.0000 | +1.4e-3 / 35 |
| 0 (root, float64 ref.) | 97,923 | 0.217 | 187.39272 | 480 | +4.6e-7 / 482 / 1.0000 | +3.5e-5 / 507 |
| 3 (float64 ref.) | 119,522 | 0.018 | 52.85712 | 245 | +3.0e-6 / 246 / 1.0000 | +5.9e-5 / 256 |
| 7 (float64 ref.) | 117,845 | 0.006 | 27.72475 | 175 | +1.4e-6 / 176 / 1.0000 | +2.4e-5 / 182 |

The float64 reference itself matches LIBLINEAR at tol 1e-8 on node 45 (gap -1.9e-9, same 33 nonzeros).

At LIBLINEAR's own production settings (tol 1e-4, 200 iterations) on 14 nodes of 2k to 59k rows, ours at 1e-4 and LIBLINEAR land within about 0.5% of each other in objective (-5.0e-3 to +3.5e-3, routing agreement >= 0.9987), but LIBLINEAR itself is 0.1 to 4.4% above the true optimum there. That stop point is path-dependent, so the optimum is the reproducible target, not LIBLINEAR at 1e-4.

**Float32 floor.** Below about 1e-6 the L-BFGS-B line search hits float32 precision and the fit ends `stop_reason: "abnormal"` (not a failure: it cannot get closer at this precision). Most fits of a real run end this way (45 of 63 in the first smoke iteration). The node log records `stop_ratio`: 1 or less means the stop target was reached, R means the fit stopped at R times the target. A fit at ratio R satisfies LIBLINEAR's own stop rule at tolerance R x 1e-6, so even ratio 186 (1.9e-4) is about sklearn's default 1e-4, which the creators' code and the old trees used. 1e-7 or 1e-8 change nothing on the nodes tested, and a float64 evaluation would cost about 6x per pass (0.57 s vs 0.08 s on the root).

Verified against a float64 solve of the same problem (gap = objective above the float64 result; nonzero counts; routing-sign agreement):
- **Cold fits (iteration 1), up to ratio 26.8:** nodes 0, 3, 7 against the float64 optimum (table above), gap <= 3e-6, nonzeros within 0.6%. Three more nodes at ratio 98.9 (20 rows), 39.5 (6,618 rows) and 29.3 (220 rows): gap <= 5e-7, identical nonzeros.
- **Warm fits (iteration 2), up to ratio 319:** 11 fits at ratio 30 to 319, 119 to 127,966 rows (including the 128k-row node at ratio 94 and a 32k-row node at ratio 186), each re-solved in float64 from our answer. Worst gap 3.0e-6 (for two fits whose float64 polish hit its 600-iteration cap the gap is a lower bound; both pass), nonzeros differ by at most 4 (of 587), sign agreement 1.0 on all.
- **Why ratios run high:** the stop target scales with min(pos, neg)/n, so heavily imbalanced nodes (minority fraction 5e-5 to 8e-4 on the worst large fits) have targets float32 cannot reach; very small nodes (a few hundred rows) show it too.
- **Not covered:** abnormal fits of iterations 3 and later. The float64 check was done on iterations 1 and 2 of the smoke run only. The log keeps `stop_ratio` for every fit, so a later iteration with ratios far above these can be spot-checked from the history.

**Cost.** The evaluation is memory-bound (about 48 GB/s, near this machine's DDR5 ceiling), so the solver runs 6 BLAS threads (`threadpoolctl`, only inside the solve) on 1024-row chunks: 62 ms per evaluation on a 120k-row node against 112 ms with all 20 threads and 2048-row chunks. A cold 1e-6 solve on that node takes 64 s (was 114 s), same quality (gap +1.5e-6, 247 nonzeros against the optimum's 245). Warm start (the node's current weights, from the second TAO iteration on) reaches the same optimum; its time saving was not measured separately.

## 7. Smoke train and your next steps

**Smoke run** (new solver on the stale export, 2 TAO iterations, output kept out of `checkpoints/symbolic/`). In brackets: the same iteration of the old `checkpoints/symbolic/chosen.pt` (LIBLINEAR, 120k-row subsample cap per node, tol 1e-4):

| | iteration 1 | iteration 2 |
|---|---|---|
| time | 8m00s | 10m34s |
| objective | 104,899 (104,973) | 48,825 (48,822) |
| train mimic accuracy | 0.9502 (0.9505) | 0.9610 (0.9599) |
| nonzero weights | 7,982 (8,163) | 5,059 (5,028) |
| active nodes | 54 (55) | 44 (46) |

Peak anonymous RAM 2.35 GB (largest node batch 1.98 GB), 0 fits ended at the iteration cap. The new solver follows the old TAO trajectory to within 0.1% on objective, with a similar tree. The old run stopped at iteration 8 (objective identical at iterations 7 and 8), each iteration 371 to 516 s.

**Expect for the real run.** About 1.4 h if it converges near iteration 8 like the old run (480 s + 7 x 634 s), about 2.6 h if all 15 iterations run. The re-export is not included. If iteration 2 takes more than about 16 min, expect more than 2 h even if it stops at 8. Keep the machine otherwise idle: the solver is memory-bound, so other jobs slow it directly. A GPU solver was considered and dropped (about 2 h is acceptable).

**New outputs.** The live display shows the current node, its solver iterations and stop ratio; the TAO line per iteration lists accepted/rejected nodes and how many fits stopped converged/abnormal. `history[i]["nodes"]` keeps the per-node log; `*_metrics.json` also holds `node_sample_counts` and `leaf_class_counts`. Old checkpoints and summaries (including `chosen.pt`) lack these, so the new notebook cells (tree diagram, dashboard) fail with a KeyError if you load them through `use_checkpoint`. Retrain first.

**Your steps.**
1. Delete `checkpoints/symbolic/teacher_trainval_new*` and `teacher_test_new*` (both `.pt` and folders). The neg_ratio subset cache lives inside `teacher_trainval_new/`, so this clears it too.
2. Run notebook 03: it re-exports from `neuro_chosen.pt`, trains, prunes and fits the margin exponents (§8; the fit adds a few minutes and prints γ per node and any `WARNING: defect leaf`). The row count may differ from 437,457 with the new detector.
3. Copy the new `runN.pt` to `checkpoints/symbolic/chosen.pt` (approved overwrite; the old `chosen.pt` no longer loads, §8). Then run notebooks 04, 05, 06.
4. After you accept notebook 06, delete `checkpoints/neuro/neuro_chosen_v1.*` and `checkpoints/symbolic/symbolic_chosen_d6_l120_a0,15*`.

**Not run.** Notebooks 03 to 06 end to end. Notebook 03's new cells were run against the smoke checkpoint and the notebook JSON validates.

## 8. γ-weighted routing margin and the recall gap (added 2026-10-25)

**Diagnosis** (old draft, COCO at 640 on the old detector, so re-verify on the new run). NeSy recall 0.973 against FRCNN 0.982: FN 65 against 46, the extra 19 on mouse_bite, spur, spurious_copper and pinhole (+4 to +6 each). With every NeSy score set to 1 recall is 0.983, so the tree routes those defects to the right class and box; they are lost because the score falls under the P/R cutoff (`precision_score_threshold: 0.5`, `configs/neuro_train.yaml`). The score was `Π σ(|m_j|)` over up to six path nodes with equal weight. `short` FN is 20 for both models (the teacher's own ceiling). The draft's explanation (FN from background-heavy training, line ~1890) does not fit this.

**Method.** `score = Π_{j on path, active} c_j^γ_j`, `c_j = σ(|w_j·x + b_j|)`, `γ_j ≥ 0`, one γ per internal node shared by all classes. γ is **not** the SODT's `w`/`b`, which are never touched. γ is fitted **after TAO and pruning, tree frozen**: log-binomial regression (binomial GLM, log link, no intercept) of the teacher's softmax probability for the leaf's class, on the defect-routed rows of the full trainval export, by bounded L-BFGS-B (convex in γ, so one optimum). All γ = 1 is the old score exactly. Routing, leaf labels, decision paths and heatmaps are unchanged; only the detection score changes.

**Pre-registered (decided with the user before seeing results):** the γ-weighted margin is the NeSy main row in every table; plain margin (γ = 1) and no margin are ablation rows. γ has no hyperparameters. Expect recall to rise toward FRCNN's and precision to fall toward it; the "P/F1 beat FRCNN" line may not survive, report it as measured.

**Code.**
- `symbolic/sodt.py`: `margin_exponents` (required key in `to_state_dict`/`from_state_dict`, no fallback), `path_log_certainties`, confidence = `exp(X·γ)`.
- `symbolic/train.py`: `solve_margin_exponents` (pure fit), `fit_margin_exponents` (writes `tree_state` and `training_config["margin_fit"]` into the checkpoint and `margin_exponents` into the summary json). `symbolic/dataset.py`: the bundle now carries `teacher_scores`.
- `neurosym/inference.py::load_neurosymbolic_detector` refuses a checkpoint without `training_config["margin_fit"]`, so an unfitted tree cannot pass as the weighted row.
- `neurosym/evaluation.py::breakdown_nesy_misses`: for every ground truth FRCNN finds at the P/R operating point and NeSy does not, the cause (`score` under the cutoff, wrong `class`, `none`) and, for `score` misses, the depth of the weakest node (lowest γ·log σ).
- Notebook 03: fit cell after pruning. Notebook 06: new plain-vs-weighted-vs-off cell (per-class FN, confusion matrices, miss breakdown, fitted γ); its outputs were cleared (approved) and a stale comment fixed. `neurosym/visualization.py`: the per-node panel (Gambar 4.13) shows γ and σ^γ.
- Tests: `tests/test_margin_exponents.py`, `tests/test_miss_breakdown.py` (76 pass in total).

**Verified by smoke only:** one-image run at size 800 (P2′ 200×200, overlay mass inside the proposal box 1.0, GT-to-grid ok); `fit_margin_exponents` on 50k rows of the stale export (BCE 0.482 to 0.368, many γ at 0); the loader guard both ways; the miss breakdown on 5 images with a random tree. Not run: the notebooks themselves.

**Citations ("dasar"), verified 2026-10-25 by the main agent** (PubMed, Europe PMC, PMC, R docs):
- Path product = soft-tree path probability: Kontschieder et al. 2015, ICCV, pp. 1467–1475 (already in the draft).
- GLM with log link: McCullagh & Nelder 1989, *Generalized Linear Models*, 2nd ed., Chapman & Hall (book verified; the section on link choice was not checked); R `family` docs list `log` among the binomial links; the "log link = multiplicative effects" statement is in arXiv:2301.03349.
- Binomial regression for risk ratios: Wacholder 1986, *Am. J. Epidemiol.* 123(1):174–184 (abstract says "different link functions"; the log link comes from secondary sources).
- Convergence caveat: Williamson, Eliasziw & Fick 2013, *Emerg. Themes Epidemiol.* 10:14. The boundary (fitted probability 1 under the log link) is **one** cause of failed convergence, not the only one.
- Probability as target: Papke & Wooldridge 1996, *J. Appl. Econometrics* 11(6):619–632.
- Post-hoc fit with the model frozen: Platt (book 2000, preprint 1999; the draft says 1999, pick one), Zadrozny & Elkan 2001 ICML, Niculescu-Mizil & Caruana 2005 ICML. Exponents on a product of experts, fitted by cross-entropy: Cao & Fleet 2014 (arXiv:1410.7827), Heskes 1998 (NIPS 10, pp. 266–272), Genest & Zidek 1986. Distillation target: Hinton et al. 2015.
- **Do not** cite McNutt et al. 2003 for the link choice: its abstract never names log-binomial or a log link.
- State honestly: no published paper fits per-node exponents on a decision tree's path product; this combination is the thesis's adaptation. Unlike a classic log opinion pool, γ has no renormalization and no sum-to-1 constraint. Unlike temperature scaling, γ can change the ranking of detections (so mAP can move either way) and γ = 0 removes a node's doubt entirely.

**Open points for the thesis.**
- γ also feeds the explanation metrics: `neurosym/evaluation.py` (`routing_confidence`, used by the deletion/insertion AUC) calls `tree.route`, so it is now the γ-weighted score. Keeping it ("the same score the detector reports") is defensible, but Tabel 4.10 AUC values change meaning; say so in the text.
- The fit is in-sample: the teacher trained on the trainval images, so its probabilities there are probably more confident than on test, which pushes γ toward 0. Platt fits on held-out data for this reason. List it under limitations.
- Stale in the draft (the user rewrites, Claude does not): Tabel 4.2 and 4.5 to 4.9, Gambar 4.7 and 4.9 to 4.13 (4.13 now shows γ), the RoI statistics, the line-1890 explanation, Eq. 3.3 (becomes the γ-weighted product).
