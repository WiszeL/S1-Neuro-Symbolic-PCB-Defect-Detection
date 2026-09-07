# What I Did: Faster R-CNN + SODT for Accountable PCB Defect Detection

*Chronological account: problem, build, breakage, fixes, audit corrections. Detection numbers below predate the §6 fixes — regenerate before citing (see §7).*

## 1. Goal

A black-box detector gives a box and a score, but no answer to "why" — nothing to audit, nothing tracing a decision back to its cause. This work fixes that without giving up accuracy: keep Faster R-CNN for features and proposals, replace its classification head with a sparse oblique decision tree (SODT) trained to mimic it. Every tree decision is an exact, inspectable root-to-leaf path — real hyperplane weights over real pooled features, not a post-hoc guess.

## 2. Teacher: Faster R-CNN + SF-PSPyramid (Fung et al., 2024)

Rebuilt from Fung et al.'s non-referential DeepPCB detector:

- ResNet-50 backbone (C2–C5), SF-PSPyramid neck (no lateral Ci→Pi links — every level feeds from deeper stages only), pixel-shuffle upsampling, SKNet-style SF attention between levels, multi-scale training, L1 regression loss on both heads, Soft-NMS.

Two disclosed deviations:

| Deviation | Why |
|---|---|
| Neck 64ch, not 256 | Cross-stage call: fewer weights per tree node from the same data keeps fits well-posed and splits sparse. Cost: absolute AP not comparable to Fung Tables 1/3/4. Architecture (topology, losses, Soft-NMS, multi-scale) still faithful. |
| 15 epochs, not 12 | Disclosure only; a retrain to shave 3 epochs wasn't worth it. |

- SF attention bottleneck `z=32` is our assumption (Fung never sizes it; unrelated to Kairgeldin's `z`).

## 3. Student: SODT + TAO (Hada et al., 2024; Kairgeldin et al., 2025)

- The head becomes a tree trained with Tree Alternating Optimization: reverse-BFS node updates, each node an L1-logistic fit (LIBLINEAR) on a routing problem, leaves set to the (optionally weighted) majority label.
- Kairgeldin's sparsity exponent implemented exactly: `λ · |R_i|^α`.
- **Sparsity schedule (`l1_lambda: 20.0`, `sparsity_alpha: 0.15`) — visualization-driven.** High α scatters node weights across the grid, so heatmaps diffuse instead of sitting on defects. Low α keeps heatmaps dense on defects; l1 raised to compensate so splits stay sparse and oblique. A claim about the heatmaps, not the features.
- **Teacher-student export, audited end to end** (does held-out evaluation secretly use truth boxes? No):
  - Proposals come from the inference-path RPN in eval mode — no truth boxes injected (`proposal_source: "rpn_pre_detector_postprocess"`).
  - Features are RoI Align on those same proposals (`64×7×7`) — the identical cut inference consumes.
  - Labels are the teacher's own argmax on those RoIs, not truth. Truth rides along only as side data (`matched_gt_boxes`, `gt_iou`, `has_matched_gt`) for spatial metrics, matched after extraction.
  - Split hygiene: held-out dump from `test.txt` (500 images), disjoint from the `trainval.txt` training dump.
- Honest caveat: test proposals come from the same RPN/backbone the teacher trained — unavoidable, since the hybrid keeps them and swaps only the classifier. Mimic scores are conditioned on that deployment distribution, which is the one that matters.

## 4. Breakage: mAP stuck at 0.877

Early hybrid tied the teacher per-decision but scored mAP@0.5 = 0.877, precision = 0.847.

- **Cause: score quantization.** A tree routes each RoI to exactly one leaf, and the pruned depth-6 tree has 64 leaves with only a handful carrying defects (`open`/`mouse_bite`: one leaf each, `short`: two). Every `open` detection in the test set got the literally identical score (leaf purity). Two systems broke on the ties:
  - **AP is a ranking metric** — true/false positives can't separate; per-class AP collapses to one operating point.
  - **Soft-NMS needs ordering** — survivor picks among overlaps turned arbitrary, sometimes killing the well-localized box.

## 5. Fix: two mechanisms

**Routing-margin scoring (inference) — the main fix:**

$$\text{score}(x) = p_{\text{leaf}}(c) \times \prod_{i \,\in\, \text{path}(x),\; w_i \neq 0} \sigma\!\left(\lvert w_i^\top x + b_i \rvert\right)$$

- Product over *active* path nodes only (pruned all-zero nodes skipped — they'd shrink every sample equally). $p_{\text{leaf}}(c)$ is leaf purity.
- Each factor is that node's routing reliability; the product is a conjunction — a prediction is only as trustworthy as its weakest routing call.
- **Changes zero decisions.** Same path, leaf, label, heatmaps — only the attached confidence turns continuous. Everything read off the tree itself: no neural head, no calibrator, no teacher peeking.
- Alone (before class weighting): mAP@0.5 0.877 → 0.968, precision 0.847 → 0.912, background FPs −20%, recall flat.

**Class weighting (training):** misrouting costs more for weak-recall classes — `short` 2.0×, `spur`/`open` 1.5×, `pinhole` 1.25× — in every node's problem and in the leaf argmax. Hyperplanes lean away from background without touching its weight (which would hand back the FP gains).

**Not in the final model:** teacher-confidence weighting (downweighting samples by teacher softmax). Built, tested, removed — the promoted checkpoint never enabled it, and a 64-leaf sparse tree can't fit teacher label noise anyway, so it competed with capacity for the same job. Two mechanisms, not three (see §6).

## 6. Audit

Review against the papers and this repo's own claims found five indefensible spots. Corrections stay visible here.

1. **Faithfulness metrics were tautological.** Symbolic side zeroed features *outside* the tree's own active path — routing provably unchanged by construction, so `sufficiency = 1.000` was a theorem, not a measurement. Grad-CAM perturbed whole 7×7 cells on a different model. A proof next to an experiment.
   - **Fix:** identical unit both sides (one grid cell, all channels, same budget/schedule), each probing *its own* model. Plus an independent Grad-CAM insertion bug (step 0 hardcoded `0.0` instead of real all-zero confidence). The structural property survives as a stated mechanism property, not a table number.
   - **Superseded (SODT side only):** "identical unit" meant the same coarse 7×7 grid on both sides — fair between methods, but it shrinks SODT's finer FPN-native map to 7×7 before scoring it, handicapping the map that actually ships. Current policy: identical *budget fraction* (0.5), each method masked at its own resolution — `neurosym/evaluation.py::evaluate_faithfulness_fpn_masking`. `leaf_only` (the 7×7 SODT ranking) dropped from reported metrics; the 7×7 protocol survives only for sufficiency, which isn't discriminative either way (§8).
2. **Hyperparameters were picked against `test.txt`.** **Fix:** `tree_depth`, `l1_lambda`, `sparsity_alpha`, `class_weights` fixed up front in `configs/symbolic_train.yaml` — single source of truth, nothing overrides it, `test.txt` touched once at the end. Matches both papers' practice (Hada depth 6/5, Kairgeldin depth 5).
   - A middle version ran an 80/20 val sweep instead — removed (~3h per TAO run for four numbers the papers fix anyway; shipped no model; "won on val" while shipping a retrained tree needs its own defense).
3. **Teacher-confidence weighting credited but unused.** Removed (see §5).
4. **Spatial metrics saturated over mismatched populations.** Tight proposals cover most of the grid, so even a uniform-random heatmap scores pointing ≈0.91 — SODT's 0.93 was chance-level; Grad-CAM's 0.99 was its blurry `layer4` field, not localization. Sides also filtered different RoI sets.
   - **Fix:** `util/heatmap_metrics.py` adds a random baseline on the identical population, one shared `min_proposal_iou` filter, and a stratified split (GT < 50% of grid) where the metrics still discriminate. `notebooks/06` reports all three + the subset.
5. **Node heatmaps miss the defect, even when right.** Confirmed as representation property, not a bug — three independent checks:
   - (a) GT-to-grid projection orientation verified directly (top-left → low indices, right-half → high columns).
   - (b) Raw pooled activation alone barely beats random on loose RoIs (pointing ≈0.45 vs ≈0.43).
   - (c) Weight-lattice reshaped onto the grid peaks near random regardless of aggregation (≈0.35–0.45).
   - **Root cause, measured:** one 7×7 cell's effective receptive field is ≈**360×350 px** vs a mean proposal of ≈**33×29 px** (**12–13× the whole box**). All 49 cells read nearly the same window (cell cosine ≈0.5). The tree separates classes by *which channels fire*, not where — yet permuting cells per-sample collapses mimic macro-F1 0.884 → 0.402, so layout matters without locating the defect. Both true, no conflict.
   - Response: the exact path attribution map (§8), not a fix to the old one.

## 7. Results

**Regenerate before citing.** The up-front config (§6.2) and TAO fidelity fixes haven't been re-run; below is the last pre-audit snapshot.

| Metric | Faster R-CNN | NeSy (FRCNN + SODT) |
|---|---|---|
| mAP@0.5:0.95 | 0.755 | 0.755 *(pre-audit)* |
| mAP@0.5 | 0.980 | 0.972 *(pre-audit)* |
| Precision | 0.897 | 0.899 *(pre-audit)* |
| Recall | 0.979 | 0.977 *(pre-audit)* |
| F1 | 0.936 | 0.937 *(pre-audit)* |

- Faithfulness/spatial numbers omitted — pre-audit values came from the broken §6.1 protocol. §8 quotes dev references from the current `run1.pt`/`NEWBEST` pair; regenerate via `notebooks/03` → `notebooks/06` alongside this table.
- **Two pre-audit findings still hold** (independent of protocol and hyperparameter choice):
  - **Mimic caps at ~97% from teacher noise, not tree capacity.** Agreement by teacher confidence: <0.7 (≈4% of RoIs) → 75.5%; ≥0.99 (≈76%) → 99.98%. 93% of disagreements sit below teacher confidence 0.9 — boundary boxes near the 0.5-IoU line where argmax is near-coin-flip. No student reproduces coin flips; deeper trees didn't help.
  - **The student sometimes beats the teacher on truth while mimicking it.** No contradiction: mimic scores against teacher labels (every disagreement "loses"), detection scores against truth — and disagreements bunch where the teacher is near-random. The sparse tree fits a smoothed boundary through the noise (classic distillation denoising, helped by GT-aligned weighting and routing-margin ordering).

## 8. Claims and limits

Core claim is *accountability*: every detection ships a complete, reproducible record — which node tested what, which RoI region it weighted, what margin it crossed, which leaf it landed in. Rerun and the record is identical: no gradient estimate, no sampling. No black box offers this at any accuracy.

| Question | Answered? |
|---|---|
| Why *this* label for *this* detection? | **Yes** — exact path, region, margin, reproducible |
| What does the model use for class X overall? | **Yes** — sparse global weights cover every RoI |
| Where is it looking on the whole board? | **No** — heatmap stops at the proposal box. Grad-CAM sees the full image here; genuine edge, conceded. |

- **Receptive-field caveat, measured.** The grid-to-image projection is positionally exact (RoI Align defines it; Hada/Kairgeldin must reconstruct theirs). But each cell's *value* comes from units seeing far more than one bin: one `p2` cell traces to **361×349 px** vs a **33×29 px** mean box. So maps are **region-level** ("this node weighted this part of the RoI"), and the region is closer to the whole RoI than to one bin. This is why §6.4 saturates and why sub-bin localization is unresolvable on the pooled grid — for any method.
- **Exact path attribution** (`neurosym/heatmap.py::compute_exact_attribution`) — the answer to §6.5. RoI-Align is linear (bilinear sample + average, no ReLU), so the path score splits onto the source FPN pixels with **no approximation**: `score = Σ_p Σ_c FPN[c,p] · ∂score/∂FPN[c,p]`, coefficients read off with autograd (backbone never in graph).
  - **Exact:** the per-pixel *score*. Checked: `|Σ(map) − score| < 1e-5` over ~1900 RoIs. Nothing discarded — sign, pattern, activation all carried.
  - **Display choice, not exact:** `abs().sum(0)` to 2-D; FPN pixels, not image pixels. Still region-level.
  - **Localization** (low-coverage subset; dev refs, regenerate with §7): exact **0.48 / 0.25**, random 0.27 / 0.20. Above chance, but Grad-CAM still leads here (≈0.61 / 0.32) — conceded; localization isn't the thesis scope. (`leaf_only` dropped from this comparison — see §6.1 amendment.)
  - **Faithfulness with controls** (FPN masking, re-pool; n=960, dev refs): exact **0.876**, random 0.017. Controls: `activation_only` 0.072 (not just bright pixels); `shuffled_w` 0.651 (weight *structure* carries the signal, McNemar p ≈ 1e-37). Not beaten: `foreign_exact` ≈0.86 (p = 0.13) — can't separate one path from another, likely because all share the root in a depth-6 tree. Bounds what the number proves, not the exactness identity (decision-specific by construction).
  - **Caption rule:** panels say "exact attribution" — fair, since removing top-ranked pixels flips the prediction 0.876 vs 0.017 random. Still **not** claimed: pixel precision, or uniqueness to one path (see `foreign_exact`).
- **Per-node faithfulness** (`evaluate_faithfulness_fpn_masking`, `return["node"]`) — the six-step claim, made falsifiable per step, not just for the path as a whole: for each node on the path, mask *that node's own* map, re-pool, check whether *that node's own* routing sign flips, plus a deletion/insertion AUC over that node's routing confidence `sigmoid(|w_i·x+b_i|)` (the same quantity `NeuroSymbolicDetector` uses for routing-margin scoring). Pruned nodes (all-zero weights) are excluded — masking can't change a score that's already just the bias. Path-level and per-node numbers are **not the same measurement** (path masks the whole map and checks the *final label*; per-node masks one node's map and checks *that node's own sign*) — never report one as a proxy or average of the other.
  - Also reported: mean cosine similarity between consecutive nodes' maps, on the same population — a low number is the falsifiable form of "each step weighs a different region"; a high one would mean the six-step narrative has nothing to show.
  - **Necessity ceiling, `exact` only** (`return["node_necessity_ceiling"]`): if masking the *whole* box collapses the score to the node's bias alone, flip needs `sign(bias) != sign(score)` — a ceiling `exact`'s ranking can reach once its budget covers the node's whole nonzero support. **Not a bound for `random`**: a partial random subset can flip via unrelated cancellation even where this ceiling is 0 — checked directly on a dense-weight synthetic tree, where it did. Compare `exact`'s flip rate against this ceiling, never against `random`'s — and check `support_fraction` (`return["node"]["exact"][depth]`, the share of the box with any contribution) first: the ceiling is only tight when that's ≤ 0.5 (the masking budget); above it, `exact`'s 50% doesn't reach the whole support and the ceiling doesn't bind.
  - **Numbers pending a checkpoint re-run** — `run1.pt`/`run2.pt` aren't in the current tree (see §7). Regenerate via `notebooks/04` or `notebooks/06` before citing.
  - **Not measured, deliberately**: per-node pointing/IoU. That would test whether the region a node weighs sits on the defect — a localization claim this thesis doesn't make (see the table above). Per-node faithfulness tests a different, narrower thing: whether that region decided *that node's own question* — which is what the panels actually claim.

## 9. Closing

Interpretability cost nothing here: the hybrid ties the teacher on mAP@0.5:0.95 and beat it pre-audit on precision/F1 — the numbers a QC operator trusts a flagged board by. "It's less accurate" is off the table. The remaining question — not "is it interpretable" (true by construction) but "is what it shows actually true" — is what this document tried to answer honestly: three assumed-true spots found and fixed, limits stated where they are.
