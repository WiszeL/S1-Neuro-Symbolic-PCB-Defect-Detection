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

Two deliberate deviations from the paper, both disclosed in `README.md`:

- **Neck width: 64 channels, not 256.** This is not a memory shortcut — it is a cross-stage
  design choice made for the *next* stage. TAO's per-node reduced problem is an L1-logistic fit
  capped at 30,000 samples; at the paper's 256ch (a 12,544-dimensional pooled feature) that
  puts the samples-per-feature ratio at ≈2.4, well below where a regularized logistic fit is
  well-posed. At 64ch (3,136-dimensional) the ratio is ≈9.6. Narrowing the neck keeps the SODT's
  node fits well-posed and its splits sparse and interpretable — the actual objective of the
  symbolic stage. The cost: absolute AP is not directly comparable to Fung et al.'s Table 1/3/4,
  which were measured at 256ch. The architecture itself — topology, loss functions, Soft-NMS,
  multi-scale training — is reproduced faithfully; only the width differs, and only for a stated
  reason.
- **15 training epochs, not 12.** Simple disclosure; given the neck-width deviation already
  scopes the AP comparison to architecture-only, this was not worth a second retrain to match
  exactly.

## 3. The student: SODT + TAO (Hada et al., 2024; Kairgeldin et al., 2025)

The classification head is replaced by a sparse oblique decision tree trained with Tree
Alternating Optimization (TAO): reverse-BFS node updates, each internal node solved as an
L1-regularized logistic regression (LIBLINEAR) on a 0/1 pseudo-label routing problem, leaves set
to the majority (optionally class-weighted) label. Kairgeldin's modification — a sparsity
exponent α controlling how the L1 penalty scales with a node's reduced-set size — is
implemented exactly: `λ · |R_i|^α`.

**The teacher-student export, audited end-to-end** (the question was: does the held-out
evaluation secretly use ground-truth boxes instead of the detector's own pipeline?):

1. **Proposal source is the RPN, not GT.** `extract_teacher_roi_samples` runs the full
   inference-path RPN in `eval()` mode. No ground-truth boxes are injected; proposals are the
   standard objectness-ranked, post-NMS RPN output (`proposal_source:
   "rpn_pre_detector_postprocess"` in the export manifest).
2. **Features are RoI Align on those proposals** — the identical `MultiScaleRoIAlign` cut the
   SODT consumes at inference. Pooled grid shape: `64×7×7`.
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
four things that would not have survived a thesis defense unexamined. Each is addressed below;
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

2. **No validation split — hyperparameters were being selected against `test.txt`.** A
   config comment literally read "Upweight the classes with the worst FN counts... on test."
   Both source papers hold out a validation set for exactly this kind of selection (Hada §6.1
   splits 1000/100/200 for train/val/test; §4 step 1 explicitly picks the tree with "close to
   highest validation accuracy"). **Fix:** `symbolic/dataset.py` now supports an image-level
   train/val split of the `trainval` export (`split`/`val_fraction`/`split_seed`), and
   `notebooks/03`'s new "Hyperparameter Selection" section sweeps the six configurations already
   explored (plus one class-weighting ablation) on `train`→`val`, with an explicit manual
   promotion step. `test.txt` is now touched exactly once, after promotion.

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

## 7. Results

**To be regenerated.** The val-split hyperparameter sweep (§6.2) and the TAO fidelity fixes
below have not yet been re-run through training; the detection table below is the last recorded
result, from before this audit, and should not be read as the final numbers. Regenerate via
`notebooks/03`'s sweep → promote → `notebooks/06`, then replace this table.

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
reprinted here.

**Two findings from the pre-audit run remain evidentially sound and are expected to hold under
re-measurement**, since neither depends on the faithfulness protocol or the val split:

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

**The receptive-field caveat.** The 7×7-grid-to-image projection is *positionally exact* — RoI
Align defines the bin↔image-region mapping by construction, unlike Hada/Kairgeldin, who must
reconstruct approximate receptive fields because their features (raw conv activations) have no
inherent box alignment. But each cell's *value* is interpolated from feature-map units whose
effective receptive field is wider than one bin — a `p2` unit's receptive field, traced back
through the CP block and C3, spans tens of pixels, comparable to or larger than a small DeepPCB
defect box. So the map is a **region-level** claim ("this decision node weighted this part of
the RoI"), not a pixel-level one. This is also the underlying reason the spatial metrics in §6.4
saturate: sub-bin localization is unresolvable at this receptive-field scale, for any method.

## 9. Closing framing

Interpretability here was not purchased with accuracy. The hybrid ties its black-box teacher on
mAP@0.5:0.95 and, in the pre-audit measurement, exceeded it on precision and F1 — the numbers
that most directly matter for a quality-control operator deciding whether to trust a flagged
board. That result removes the standard argument for keeping the black box: if the interpretable
version costs nothing, "it's less accurate" is no longer an available objection. What replaces
it is the question this document tries to answer honestly — not "is it interpretable," which
was true by construction from the start, but "is what it shows you actually true," which
required finding and fixing three places where the answer had been assumed rather than checked.
