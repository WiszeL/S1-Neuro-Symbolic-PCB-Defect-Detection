# What I Did: Faster R-CNN + SODT for Accountable PCB Defect Detection

*A chronological account: the problem, what was built, what broke, how it was diagnosed and
fixed, and — after a code review — what was found wrong and corrected. Numbers below are
from the DeepPCB test split. Detection numbers predate the fixes in §6 and must be
regenerated (see §7); this is stated explicitly rather than silently updated.*

---

## 1. The goal

A black-box detector gives a box and a score. When it is wrong, there is no answer to "why" —
no record a human can audit, no way to trace a specific decision back to a specific cause.
That is the problem this work addresses: build a detector whose every decision comes with a
complete, reproducible record, without giving up the accuracy that makes black-box detectors
attractive in the first place. Interpretability here is meant to be a property of the system,
not a trade-off paid for with worse detection.

The approach: keep a standard two-stage detector (Faster R-CNN) for feature extraction and
region proposal, and replace its classification head with a sparse oblique decision tree (SODT)
trained to mimic it. The tree's decision is exact and inspectable by construction — a root-to-leaf
path with real hyperplane weights over real pooled features, not a post-hoc approximation of
what the network "might have" used.

## 2. The teacher: Faster R-CNN + SF-PSPyramid (Fung et al., 2024)

Reconstructed from Fung et al.'s non-referential DeepPCB detector: ResNet-50 backbone (C2–C5,
C1 unused), the SF-PSPyramid neck (no lateral Ci→Pi connections — every Pi level is fed only
from the *deeper* Cj, forcing the model to route through semantic features rather than shallow
ones), pixel-shuffle upsampling in each CP block, SF attention (SKNet-style) fusing adjacent
pyramid levels, multi-scale training, L1 regression loss on both the RPN and RoI heads, and
Soft-NMS at inference.

One deliberate deviation from the paper, disclosed in `README.md`:

- **15 training epochs, not 12.** Simple disclosure; a second retrain to shave three epochs was
  not judged worth it.

Fung et al.'s SF attention (§3.3) compresses the pooled descriptor to `z` channels before
expanding it back — their symbol, but they never give `z` a value. It is set to 64 here as an
assumption, not a deviation. (Unrelated to Kairgeldin's `z`, the feature vector fed to the
tree.)

**Neck width: 256 channels, matching the paper.** An earlier version of this work ran the
SF-PSPyramid at 128 channels, on the reasoning that the *next* stage's TAO node fits — an
L1-logistic regression per internal node — would be better-posed on a narrower pooled feature:
at 256ch the pooled grid is 12,544-dimensional, and against the then-uncapped solver that put
the samples-per-feature ratio near the edge of where a regularized logistic fit behaves. That
reasoning was retired once the TAO solver gained an explicit sample cap (see §3): the cap fixes
the well-posedness concern directly, at the node level, without paying for it with a
paper-divergent architecture. The neck is now 256ch as Fung et al. specify, so absolute AP is
comparable to their Table 1/3/4 up to the 15-vs-12-epoch difference.

## 3. The student: SODT + TAO (Hada et al., 2024; Kairgeldin et al., 2025)

The classification head is replaced by a sparse oblique decision tree trained with Tree
Alternating Optimization (TAO): reverse-BFS node updates, each internal node solved as an
L1-regularized logistic regression (LIBLINEAR) on a 0/1 pseudo-label routing problem, leaves set
to the majority (optionally class-weighted) label. Kairgeldin's modification — a sparsity
exponent α controlling how the L1 penalty scales with a node's reduced-set size — is
implemented exactly: `λ · |R_i|^α`.

**One deviation from Hada/Kairgeldin: a solver sample cap.** Both papers fit each internal
node's reduced problem on its full reduced set. Here the LIBLINEAR fit is capped at 120,000
samples (`symbolic/tao.py`, `solver_cap`): LIBLINEAR is double-precision and materializes the
feature matrix densely, so at 256ch an uncapped root reduced set (~350k rows on the DeepPCB
`trainval` export) needs roughly 50 GB of RAM. Only the root and the top two or three nodes
ever exceed 120k — every deeper node still fits on its full set. At the cap the
samples-per-feature ratio is ≈9.6, comfortably in the well-posed regime, and the subsample is
uniform (unbiased; the per-node regularization `C` is still computed from the true `|R_i|`).
The effect on the learned splits is expected to be within TAO's run-to-run variation, but this
has not been verified against an uncapped baseline.

**The teacher-student export, audited end-to-end** (the question was: does the held-out
evaluation secretly use ground-truth boxes instead of the detector's own pipeline?):

1. **Proposal source is the RPN, not GT.** `extract_teacher_roi_samples` runs the full
   inference-path RPN in `eval()` mode. No ground-truth boxes are injected; proposals are the
   standard objectness-ranked, post-NMS RPN output (`proposal_source:
   "rpn_pre_detector_postprocess"` in the export manifest).
2. **Features are RoI Align on those proposals** — the identical `MultiScaleRoIAlign` cut the
   SODT consumes at inference. Pooled grid shape: `256×7×7`.
3. **Labels are the teacher's own predictions** — `argmax(softmax(classifier(box_head(pooled))))`
   on the same RoIs, not ground truth. Ground truth appears only as a side channel
   (`matched_gt_boxes`, `gt_iou`, `has_matched_gt`) for spatial explanation metrics, matched by
   IoU *after* extraction, never as a mimic target.
4. **Split hygiene**: the held-out dump is built from `test.txt` (500 images), disjoint from
   the `trainval.txt` dump the tree trains on.

One honest caveat: the RoI proposals on test images come from the same RPN whose backbone the
teacher trained — by construction, since the hybrid keeps Faster R-CNN's backbone/RPN/regressor
and replaces only the classification head. The mimic metric is conditioned on that deployment
distribution, which is precisely the distribution that matters at inference.

## 4. What broke: mAP stuck at 0.877

An early hybrid tied the teacher on mimic accuracy but scored mAP@0.5 = 0.877, precision =
0.847 — far below the teacher's own detection numbers, despite the tree agreeing with the
teacher on individual routing decisions. The cause was not tree accuracy.

**Diagnosis: score quantization.** A decision tree is a hard router — every RoI lands in exactly
one leaf, and the pruned depth-6 tree has 64 leaves, of which only a handful carry defect
classes (`open` and `mouse_bite` each had exactly one leaf, `short` had two). Consequence:
*every* `open` detection in the entire test set received the literally identical score (the
leaf's purity). Two downstream systems assume scores are a meaningful ordering, and both broke
on the tie:

- **Average Precision is a ranking metric.** With one distinct score per class, true and false
  positives cannot be separated in the ranking; per-class AP degenerates to roughly the
  precision at a single operating point.
- **Soft-NMS needs score ordering to pick survivors.** With tied scores, which overlapping box
  survives suppression is arbitrary — sometimes the well-localized box lost, sometimes a
  duplicate won. This produced a confusion-matrix signature that looked like a Soft-NMS bug but
  was a *score* bug.

## 5. The fix: two mechanisms

### Routing-margin scoring (inference) — the main fix

$$\text{score}(x) = p_{\text{leaf}}(c) \times \prod_{i \,\in\, \text{path}(x),\; w_i \neq 0} \sigma\!\left(\lvert w_i^\top x + b_i \rvert\right)$$

The product runs over the *active* internal nodes on the RoI's root-to-leaf path (pruned
all-zero nodes are skipped, since their score is identically zero and would apply a uniform
shrink to every sample). $p_{\text{leaf}}(c)$ is the leaf purity for the predicted class.

Intuition: $w_i^\top x + b_i$ is the signed distance to node $i$'s hyperplane — its sign decides
routing (untouched), its magnitude is the classical margin (how far the sample sits from the
boundary). $\sigma(|\cdot|)$ turns each margin into a per-node routing reliability, and the
product over the path is a natural conjunction — a prediction is only as trustworthy as its
*least* confident routing decision.

Why this is the right kind of fix, not a hack: **it changes zero decisions.** Same path, same
leaf, same predicted label, same node heatmaps — only the confidence attached to that unchanged
decision becomes continuous. Every factor is read off the tree itself; no neural head, no
learned calibrator, no peeking at the teacher. The explanation figures already print these
node scores; the detection score is now literally a function of the numbers already shown in
the explanation.

Measured effect of this change alone (before class weighting): mAP@0.5 0.877 → 0.968,
precision 0.847 → 0.912, background false positives cut by ~20%, recall essentially unchanged.

### Class weighting (TAO training)

Misrouting a `short` RoI costs 2× (and `spur` 1.5×) in every node's reduced problem, and the
leaf-label argmax uses weighted counts. This shifts decision hyperplanes away from background
specifically for the classes with the worst false-negative counts, without touching
background's weight (which would trade the false-positive gains back).

**A third mechanism — teacher-confidence weighting (downweighting TAO samples by the teacher's
softmax confidence) — was implemented and tested but is *not* part of the final model.** The
promoted checkpoint's training config has no such setting; it was never enabled in the reported
result. It was removed after a review found it credited as a fix in an earlier draft of this
document despite not being used, and found a structural reason it likely wouldn't help: a
64-leaf, L1-sparse tree already lacks the capacity to fit teacher label noise (the actual
justification for confidence weighting), so the two mechanisms compete for the same job: See §6.
The causal chain is **two** mechanisms, not three.

## 6. The audit

A code review against the source papers and against the codebase's own academic claims found
five things that would not have survived a thesis defense unexamined. Each is addressed below;
this section exists so the correction is visible, not silent.

1. **Faithfulness metrics were tautological.** Sufficiency/necessity/deletion/insertion on the
   symbolic side zeroed only the features *outside* the tree's own active path — every node on
   that path has nonzero weight only inside that set, so the routing is provably unchanged by
   construction. `sufficiency_prediction_preservation = 1.000` was a theorem, not a measurement.
   Meanwhile Grad-CAM's faithfulness metrics perturbed whole 7×7 spatial cells (all channels) and
   probed the FRCNN box head — a different unit, a different model. Comparing the two was
   comparing a proof to an experiment.

   **Fix:** both sides now perturb the identical unit — one spatial cell of the shared pooled
   grid, all channels, ranked by each method's own heatmap, identical cell budget and step
   schedule — and each probes *its own* model (self-consistency, not cross-model transfer). See
   the module docstrings in `symbolic/evaluation.py` and `gradcam/evaluation.py`. Grad-CAM's
   insertion-AUC also had an independent bug (step 0 was left at a hardcoded `0.0` instead of the
   real all-zero-input confidence, biasing the curve low); fixed alongside.

   The structural property survives, correctly framed: because the path's nonzero-weight set
   fully determines the routing, the SODT's decision is sufficient and necessary *by
   construction*. That belongs in this document as a stated property of the mechanism, not as a
   number sitting next to Grad-CAM's in a results table.

2. **Hyperparameters were being selected against `test.txt`.** A config comment literally read
   "Upweight the classes with the worst FN counts... on test." **Fix:** `tree_depth`,
   `l1_lambda`, `sparsity_alpha`, and `class_weights` are now fixed a priori in
   `configs/symbolic_train.yaml` and disclosed there as a stated design choice — the config is
   the single source of truth, nothing overrides it, and no selection procedure runs against any
   evaluation split. This matches both source papers, which fix tree depth before training rather
   than searching it (Hada §6.1 depth 6, §6.2 depth 5; Kairgeldin §6 depth 5). `test.txt` is
   touched exactly once, at the end.

   A middle version of this codebase instead added an image-level train/val split of the
   `trainval` export and a validation sweep in `notebooks/03`. That was removed: at ~3 h per TAO
   run it cost three extra full trainings to choose four numbers the papers fix a priori, it
   shipped no model of its own (the final tree is always retrained from scratch on the full
   dump), and reporting a config as "won on val" while shipping a differently-trained tree is
   itself a thing to defend. Fixing the values a priori is the smaller claim.

3. **Teacher-confidence weighting was credited but unused.** See §5 — removed.

4. **Spatial metrics (pointing game, heatmap IoU) were saturated and measured over mismatched
   populations.** GT boxes typically cover most of the 7×7 grid (the proposal is tight on the
   defect), so a *uniform-random* heatmap scores pointing ≈0.91 — meaning the reported 0.93 for
   SODT was statistically indistinguishable from chance. Grad-CAM's 0.99 was traceable to its
   `layer4` receptive field producing a smooth, centre-weighted blur rather than genuine
   localization. Separately, the two sides were filtering different RoI populations (symbolic
   applied `min_proposal_iou=0.5`, Grad-CAM applied no such filter). **Fix:**
   `util/heatmap_metrics.py` now provides a random-heatmap baseline
   (`evaluate_random_baseline_spatial_metrics`) computed over the identical population and GT
   projection, a shared `min_proposal_iou` filter applied to both sides, and a stratified
   breakdown (`stratified_spatial_result`) reporting the subset of RoIs where the GT box covers
   less than 50% of the grid — the regime where these metrics still discriminate. `notebooks/06`
   reports all three (SODT, Grad-CAM, random) with the stratified subset alongside the overall,
   overall-saturated numbers.

5. **The node heatmap highlights areas that are not the defect, even on correct
   classifications.** Investigated as a possible bug before being confirmed as a property of the
   pooled representation. Three independent checks, each of which would catch a bug if one were
   there: (a) `project_gt_box_to_roi_grid`'s row/col orientation is correct — a GT box occupying
   the top-left quarter of the proposal projects to low row/col indices, a right-half GT projects
   to high column indices, verified directly; (b) the **raw pooled activation**, with zero tree
   math involved at all, barely beats a random heatmap on RoIs where the GT box covers less than
   half the grid (pointing ≈ 0.45 vs a random-heatmap baseline ≈ 0.43 — chosen because tight
   proposals saturate this metric near-identically for every method, masking the effect); (c)
   *reshaping the node's weight lattice back onto the 7×7 grid* — the "leaf-only" heatmap — and
   forcing a 2D peak out of it lands near that same random baseline regardless of aggregation
   formula (positive/negative/absolute/signed all ≈ 0.35–0.45).

   *(These per-cell figures come from a one-off investigation script that was not committed; the
   receptive-field measurements below are in the same category. The numbers that survive into a
   permanent, re-runnable form are the exact-attribution metrics in §8 and the "does the SODT use
   the 7×7 layout at all" probe — see §8.)*

   **Root cause, measured directly:** one 7×7 cell's effective receptive field is roughly
   **360×350 px** (gradient-traced back to the input image), while the mean proposal box is only
   about **33×29 px** — the receptive field is **12–13× larger than the entire box**. All 49
   cells therefore read almost the same window (cell-to-cell cosine similarity ≈ 0.5).
   Consequence: on the *pooled grid*, variation across cells is far smaller than variation across
   channels, and the tree's weight mass shows no preference for the cells covering the defect.
   The SODT separates classes by *which channels fire*. It also uses the 7×7 layout — permuting
   the cells independently per sample collapses held-out mimic macro-F1 from 0.884 to 0.402, so
   the layout is not noise — but that layout does not align with *where the defect is*: the two
   facts are both true and do not conflict.

   **Not a fix, a finding — see the "Exact path attribution" subsection in §8** for the map that
   replaced the channel-collapsed heatmap and what is now claimed and not claimed as a result.

## 7. Results

**To be regenerated.** The a-priori config (§6.2) and the TAO fidelity fixes below have not yet
been re-run through training; the detection table below is the last recorded result, from before
this audit, and should not be read as the final numbers. Regenerate via `notebooks/03` (train →
evaluate once on `test.txt`) → `notebooks/06`, then replace this table.

| Metric | Faster R-CNN | NeSy (FRCNN + SODT) |
|---|---|---|
| mAP@0.5:0.95 | 0.755 | 0.755 *(pre-audit)* |
| mAP@0.5 | 0.980 | 0.972 *(pre-audit)* |
| Precision | 0.897 | 0.899 *(pre-audit)* |
| Recall | 0.979 | 0.977 *(pre-audit)* |
| F1 | 0.936 | 0.937 *(pre-audit)* |

Faithfulness and spatial numbers are omitted entirely — the pre-audit values (sufficiency
1.000, necessity flip 0.999, deletion 0.260, insertion 0.812, pointing 0.930, IoU overlap
0.910) were produced by the broken protocol in §6.1 and would misrepresent the fixed one if
reprinted here. The exact-attribution figures quoted in §8 (necessity 0.876, localization
≈ 0.48 / 0.25) are dev references from the current `run1.pt`/`NEWBEST` pair; regenerate them
in the same `notebooks/06` pass that replaces this table.

**Two findings from the pre-audit run remain evidentially sound and are expected to hold under
re-measurement**, since neither depends on the faithfulness protocol or on how hyperparameters
are chosen:

- **Held-out mimic accuracy caps at ~97%, and this is teacher noise, not tree capacity.**
  Stratifying tree–teacher agreement by the teacher's own softmax confidence: at confidence
  <0.7 (≈4% of RoIs) agreement is 75.5%; at ≥0.99 (≈76% of RoIs) agreement is 99.98%. 93% of all
  disagreements occur where the teacher's own confidence is below 0.9. The uncertain RoIs are
  boundary proposals — boxes half-covering a defect near the 0.5-IoU threshold — where the
  teacher's argmax is closer to a coin flip than a decision. No student of any capacity
  reproduces coin flips; deeper trees did not help empirically. The residual gap is a property
  of the teacher's own decision boundary, not the tree's.
- **The student sometimes beats the teacher on ground truth despite mimicking it.** This is not
  a contradiction: mimic accuracy is scored against teacher labels (the tree "loses" every
  disagreement by definition), while detection metrics are scored against ground truth — and
  disagreements concentrate exactly where the teacher is near-random, so a disagreeing student
  is not penalized by reality the way it is penalized by the mimic metric. A sparse, L1-regularized
  tree cannot represent the teacher's noisy high-curvature boundary and instead fits a smoothed
  version through the ambiguous region — classic distillation denoising, reinforced by class
  weighting (GT-aligned, not teacher-aligned) and by routing-margin's effect on Soft-NMS
  ordering.

## 8. What this does and does not claim

The central thesis claim is *accountability*: every detection comes with a complete,
reproducible decision record — which node tested what, which region of the RoI it weighted,
what margin it crossed by, and which leaf it landed in. Re-run the same input and you get the
identical record; there is no gradient estimate, no sampling, nothing to reproduce
approximately. A black-box detector cannot offer this at any accuracy.

That claim has a precise scope, established directly by the mechanism:

| Question | Answered? |
|---|---|
| "Why did *this* detection get *this* label?" | **Yes** — exact path, region, and margin, reproducible |
| "What does the model use for class X in general?" | **Yes** — the tree's sparse global weights describe every RoI, not just one |
| "Where is the model looking on the whole board?" | **No** — the heatmap is confined to the detected proposal box (`neurosym/heatmap.py`'s projection is zero outside it by construction). Grad-CAM's CAM is computed on the pre-crop backbone feature map and can show attention spread across the whole image — a genuine advantage on this specific axis, conceded rather than argued away. |

**The receptive-field caveat, quantified.** The 7×7-grid-to-image projection is *positionally
exact* — RoI Align defines the bin↔image-region mapping by construction, unlike
Hada/Kairgeldin, who must reconstruct approximate receptive fields because their features (raw
conv activations) have no inherent box alignment. But each cell's *value* is interpolated from
feature-map units whose effective receptive field is far wider than one bin: gradient-traced
directly from a pooled cell back to the input image, one `p2`-level cell's receptive field
measures **361×349 px**, against a mean DeepPCB proposal box of **33×29 px** — **12–13× the
entire box**, not merely "comparable to or larger than" it. So the map is a **region-level**
claim ("this decision node weighted this part of the RoI"), not a pixel-level one, and the
region in question is closer to "the whole RoI and its surroundings" than to any one bin. This is
also the underlying reason the spatial metrics in §6.4 saturate, and the direct cause of §6.5:
sub-bin localization is unresolvable at this receptive-field scale, for any method operating on
the pooled grid.

**Exact path attribution — what it is, and what it is not.** §6.5 diagnosed *why* the
pooled-grid heatmap cannot localize; this is the response. `neurosym/heatmap.py::compute_exact_attribution`
does not weight or approximate anything. RoI-Align is linear (bilinear sampling + averaging, no
ReLU), so the SODT path's score decomposes onto the pixels of the FPN level map it was pooled
from with **no approximation**:

```
score = Σ_c Σ_ij W[c,i,j] · pooled[c,i,j] = Σ_p Σ_c FPN[c,p] · ∂score/∂FPN[c,p]
```

`∂score/∂FPN` is just RoI-Align's own linear coefficients, read off with autograd (the backbone
is never in the graph); `grad · activation`, summed over channels, is the per-pixel
contribution. The level comes from RoI-Align's `LevelMapper`, not a guess.

- **What is exact:** the whole per-pixel *score* attribution. Verified: `|Σ(map) − score|` is at
  the floating-point floor (`< 1e-5` over ~1900 RoIs). No gradients estimated, no surrogate, no
  second network, and — unlike the earlier channel-collapsed heatmap — nothing about `W`
  discarded: sign, per-cell pattern and activation are all carried.
- **What is a presentation choice, not exact:** reducing the `(C, Hf, Wf)` contribution to one
  2-D map via `abs().sum(0)`; and the resolution — exactness reaches the FPN feature map, whose
  pixels each summarise a wide receptive field, not raw image pixels. So the map is a
  *region-level* claim.
- **Localization** (pointing / IoU vs the GT box, low-GT-coverage subset; *dev reference on the
  current `run1.pt` pair, regenerate alongside §7*): exact **0.48 / 0.25**, leaf-only 0.33 /
  0.25, random 0.27 / 0.20. Paired tests: exact is statistically tied with the old channel-
  collapsed map on pointing and beats it on IoU; both the exact map and leaf-only clear random
  on one axis each, and only the exact map clears it on both. Grad-CAM still scores higher on
  localization alone (≈ 0.61 / 0.32 on the same subset) — a genuine gap on that axis, conceded
  rather than argued away. Faithfulness, not localization, is the thesis's stated scope.
- **Faithfulness, measured with controls, not asserted.** Masking directly in the FPN map and
  re-pooling (`neurosym/evaluation.py::evaluate_faithfulness_fpn_masking`), necessity flip rate
  (n=960, dev reference): **exact 0.876**, leaf-only 0.414, random 0.017. §6.1 is an audit
  finding about a metric that was true by construction, so three controls were run:
  `activation_only` (rank FPN pixels by activation magnitude, no tree) flips 0.072 — the exact
  map is not just deleting the brightest pixels; `shuffled_w` (the same weight values, entries
  permuted) flips 0.651 — the SODT's weight *structure*, not merely its scale, carries a large,
  significant share of the signal (McNemar p ≈ 1e-37). The one control it does **not** beat is
  `foreign_exact` (another RoI's path weights, ≈ 0.86, p = 0.13): this test cannot separate one
  root-to-leaf path from another, most plausibly because every path shares the root node in a
  depth-5 tree. That is a bound on what the necessity number proves — not on the exactness
  identity, which is decision-specific by construction.
- **The caption rule.** Panels are labelled "exact attribution" / "exact path attribution".
  Because the necessity test measures exactly this — removing the pixels the map ranks highest
  flips the SODT's prediction 0.876 of the time vs 0.017 for random — the panels *can* be read
  as showing what the decision used. Still **not** claimed: pixel-level precision (one FPN pixel
  summarises a wide receptive field), and uniqueness to this exact root-to-leaf path (the
  `foreign_exact` control above).

## 9. Closing framing

Interpretability here was not purchased with accuracy. The hybrid ties its black-box teacher on
mAP@0.5:0.95 and, in the pre-audit measurement, exceeded it on precision and F1 — the numbers
that most directly matter for a quality-control operator deciding whether to trust a flagged
board. That result removes the standard argument for keeping the black box: if the interpretable
version costs nothing, "it's less accurate" is no longer an available objection. What replaces
it is the question this document tries to answer honestly — not "is it interpretable," which
was true by construction from the start, but "is what it shows you actually true," which
required finding and fixing three places where the answer had been assumed rather than checked.
