# NeSy (Faster R-CNN + SODT): Findings Summary

*Why the hybrid now matches the black-box teacher, what each fix contributed, why the mimic
accuracy is capped at ~97%, why the student occasionally beats the teacher, and a verification
of the held-out mimic evaluation protocol. All numbers are from the DeepPCB test split with the
`NEWBEST` checkpoints (class-weighted SODT, routing-margin scoring).*

---

## 1. Final results

| Metric | Faster R-CNN | NeSy (FRCNN + SODT) |
|---|---|---|
| mAP@0.5:0.95 | 0.755 | **0.755** |
| mAP@0.5 | 0.980 | 0.972 |
| Precision | 0.897 | **0.899** |
| Recall | 0.979 | 0.977 |
| F1 | 0.936 | **0.937** |

The interpretable hybrid ties the black-box on COCO mAP, beats it on precision and F1, and is
within 0.008 mAP@0.5 — while keeping perfect explanation faithfulness (necessity flip rate
1.000, sufficiency preservation 1.000 vs GradCAM's 0.052 / 0.948).

For reference, before the fixes described below the hybrid stood at mAP@0.5 = 0.877 and
precision = 0.847 with an identical tree-decision quality. Every point of the recovery came
from *scoring and training-emphasis* changes, not from changing the SODT architecture, its
input (raw C×7×7 RoI Align grid), or its explanations.

## 2. The causal chain of fixes

The three mechanisms, in the order they act in the pipeline:

1. **Teacher-confidence weighting (TAO training).** Each RoI's contribution to the node-level
   L1-logistic reduced problems is scaled by the teacher's max softmax. RoIs where the teacher
   itself is unsure (boundary proposals) are soft-labels — downweighting them stops the tree
   from spending capacity fitting label noise.
2. **Class weighting (TAO training, `short: 2.0`, `spur: 1.5`).** Misrouting a `short` RoI
   costs 2× in every node's reduced problem, and the leaf-label argmax uses weighted counts.
   This shifts decision hyperplanes *away from background* exactly for the classes with the
   worst false-negative counts, without touching background's weight (which would have traded
   the false-positive gains back). Result: `short` diagonal 453 → 457 (now beating the
   teacher's own 456) with no FP regression.
3. **Routing-margin scoring (inference).** The key fix — see §3.

## 3. Routing-margin scoring — why it was *the* fix

### 3.1 The failure: score quantization

A decision tree is a hard router: every RoI lands in exactly one leaf, and the score the
detector used was that leaf's class distribution. The pruned depth-6 tree has 64 leaves, of
which only a handful carry defect classes — **`open` and `mouse_bite` each have exactly one
leaf**, `short` has two. Consequence: *every* `open` detection in the entire test set received
the literally identical score (the leaf purity, e.g. 0.942).

Two downstream systems silently assume scores are a meaningful *ordering*, and both collapsed:

- **Average Precision is a ranking metric.** AP integrates precision over detections sorted by
  score. With one distinct score per class, true and false positives cannot be separated in the
  ranking — the sort order within the tie is arbitrary — and per-class AP degenerates to
  roughly the precision at a single operating point. Measured: precision was 0.847, and
  mAP@0.5 sat at 0.877. The numbers match because they were the same quantity in disguise.
- **Soft-NMS needs score ordering to pick survivors.** When several overlapping candidate
  boxes for the same defect all carry the same score, which one decays is arbitrary: sometimes
  the well-localized box was suppressed (→ false negative), sometimes a duplicate survived
  above threshold (→ false positive). This produced the "too many FN, and some FP" confusion-
  matrix signature, which looked like a Soft-NMS bug but was a *score* bug.

### 3.2 The fix

$$\text{score}(x) \;=\; p_{\text{leaf}}(c) \times \prod_{i \,\in\, \text{path}(x),\; w_i \neq 0} \sigma\!\left(\lvert w_i^\top x + b_i \rvert\right)$$

where the product runs over the *active* internal nodes on the RoI's root-to-leaf path
(pruned all-zero nodes are skipped — their score is identically 0 and would only apply a
uniform ×0.5 shrink), and $p_{\text{leaf}}(c)$ is the leaf purity for the predicted class.

Intuition, term by term:

- $w_i^\top x + b_i$ is the signed distance of the RoI's feature vector to node $i$'s oblique
  hyperplane. Its *sign* decides left/right — that is the routing, and it is untouched.
  Its *magnitude* is the classical margin: how far the sample sits from the decision boundary.
- $\sigma(|\cdot|) \in [0.5, 1)$ converts each margin into a per-node routing reliability:
  a sample far from the hyperplane (margin 5–13, typical for clean defects) contributes
  $\approx 1$; a sample skimming the boundary (margin $\approx 0$, typical for ambiguous
  boundary proposals) contributes $\approx 0.5$.
- The product over the path is the natural conjunction: a prediction is only as trustworthy
  as its *least* confident routing decision. One near-tie anywhere on the path drags the
  score down; a path of confident splits keeps the leaf purity nearly intact.

Why this is the *right* kind of fix for the thesis, not a hack:

- **It changes zero decisions.** Same path, same leaf, same predicted label, same node
  heatmaps, same mimic accuracy — only the confidence value attached to the (unchanged)
  decision becomes continuous. The explanation story is untouched.
- **It is purely symbolic.** Every factor is read off the tree itself (the hyperplanes the
  explanations already visualize) — no neural head, no learned calibrator, no peeking at the
  teacher. The tree-path figures already print these node scores (e.g. "Score: −8.26 → went
  RIGHT"); the detection score is now literally a function of the numbers shown in the
  explanation.
- **It is monotone within a leaf**, so within-class ranking is by margin-reliability — exactly
  the signal AP and Soft-NMS were missing. Empirically the confidence tracks proposal quality
  with no supervision: mean 0.915 for proposals with GT IoU ≥ 0.7 vs 0.644 for boundary
  proposals (IoU 0.3–0.5).

Measured effect of this change alone (before class weighting): mAP@0.5 0.877 → 0.968,
precision 0.847 → 0.912, background false positives cut by ~20% (e.g. `short` FP 132 → 61),
recall essentially unchanged. Class weighting then recovered the remaining FN, landing at the
table in §1.

## 4. Why held-out mimic accuracy is capped at ~97% — and why that is a finding, not a failure

Stratify the tree–teacher agreement on the held-out test dump by the *teacher's own* softmax
confidence:

| Teacher max-softmax | Share of RoIs | Tree agreement |
|---|---|---|
| < 0.7 | ~4% | 75.5% |
| 0.7 – 0.9 | ~7% | 91.1% |
| 0.9 – 0.99 | ~13% | 98.9% |
| ≥ 0.99 | ~76% | 99.98% |

**93% of all tree–teacher disagreements occur on RoIs where the teacher's confidence is below
0.9.** On confident RoIs, per-class agreement is ≥ 98.9% for every class, including `short`
(99.8%) — the class whose headline agreement (92–96%) looked weakest simply because 46% of its
RoIs fall in the teacher-uncertain band (mean teacher confidence 0.854, the lowest of all
classes).

The interpretation: the residual ~3% is **aleatoric label noise, not tree capacity**. The
uncertain RoIs are boundary proposals — boxes half-covering a defect near the 0.5-IoU
foreground threshold — where the teacher's softmax is nearly uniform and its argmax label is
effectively a coin flip. No student of any capacity reproduces coin flips; a deeper tree would
only memorize this noise (and did not help empirically: neg_ratio sweeps up to the full ~1M-RoI
dump moved nothing). The mimic ceiling is a property of the *teacher's* decision boundary, and
the stratified table is the evidence: wherever the teacher actually knows the answer, the
64-leaf sparse tree follows it at ≥ 99.5%.

## 5. Why the student sometimes beats the teacher

The confusion matrices show the hybrid *outperforming* the raw Faster R-CNN in places
(`short` 457 vs 456 correct, background→`open` FP 24 vs 33, background→`spurious_copper`
120 vs 123 …). How can a mimic beat what it mimics? Because *mimic accuracy* and *detection
accuracy* are measured against different references:

- Mimic accuracy compares tree vs **teacher labels** — the tree "loses" every disagreement by
  definition.
- Detection metrics compare both models vs **ground truth** — and the tree's disagreements
  with the teacher are concentrated (92%) exactly where the teacher is near-random. In that
  region the teacher is right only about half the time, so a disagreeing student is *not*
  penalized by reality the way it is penalized by the mimic metric. Three mechanisms tilt
  those disagreements in the student's favor:

1. **Regularization as denoising.** A 64-leaf tree with L1-sparse oblique splits cannot
   represent the teacher's noisy, high-curvature boundary wiggles, so it fits a smoothed
   boundary through the ambiguous region. Where the teacher's boundary wobbles due to noise,
   the smoothed boundary is closer to the true class boundary — classic student-distillation
   denoising, here made stronger by confidence weighting, which explicitly told the tree to
   care less about the noisy region.
2. **Class weighting is GT-aligned, not teacher-aligned.** Upweighting `short`/`spur` pushes
   ambiguous RoIs toward "defect" — a deliberate, systematic bias that costs mimic agreement
   (the teacher said background) but wins detection recall (the ground truth says defect).
3. **Different score geometry through Soft-NMS.** Routing-margin scores rank duplicates by
   geometric margin rather than by the teacher's softmax, and empirically that ordering
   correlates with localization quality (§3.2), so suppression keeps better boxes slightly
   more often.

None of this contradicts "the student just follows the teacher": it follows the teacher
wherever the teacher is decisive (99.5%+), and where the teacher is guessing, the student's
inductive bias (sparsity, margins, class weights) guesses slightly better than the teacher's
noise.

## 6. Verification of the held-out mimic evaluation — protocol is correct

Audited end-to-end (`symbolic/export.py`, `neuro/faster_rcnn.py::extract_teacher_roi_samples`,
`symbolic/train.py::evaluate_heldout`). The concern was: "does the held-out set take RoI Align
features from ground-truth boxes instead of the Faster R-CNN pipeline?" It does not:

1. **Proposal source = the RPN, not GT.** `extract_teacher_roi_samples` runs the full
   inference-path RPN (`_extract_proposal_feature_records` → `self.rpn(...)` in `eval()` mode).
   The custom `L1RegionProposalNetwork` only overrides the *training loss*; in eval mode
   targets are ignored entirely, so proposals are the standard objectness-ranked, post-NMS RPN
   output. No ground-truth boxes are injected and no training-style GT-append/sampling is used
   (`proposal_source: "rpn_pre_detector_postprocess"` in the export payload).
2. **Features = RoI Align on those proposals.** The pooled C×7×7 grids are extracted by the
   same `MultiScaleRoIAlign` the hybrid uses at inference — the identical feature cut the SODT
   consumes in deployment.
3. **Labels = the teacher's own predictions.** `teacher_labels = argmax(softmax(classifier
   (box_head(pooled))))` on the same RoIs. Mimic accuracy therefore measures exactly what it
   claims: agreement with the teacher on the deployment feature/proposal distribution.
   Ground truth appears only in the side-channel fields (`matched_gt_boxes`, `gt_iou`,
   `has_matched_gt`), which are matched by IoU *after* extraction and used solely for spatial
   explanation metrics and stratified analysis — never as mimic targets.
4. **Split hygiene.** The held-out dump is built from `test.txt` images (500 images, ~500k
   RoIs), disjoint from the `trainval.txt` dump the tree was trained on, and
   `evaluate_heldout` loads it without `neg_ratio` subsampling (all RoIs evaluated).

One honest caveat to state in the thesis: the RoI *proposals* on test images come from the
same RPN whose backbone the teacher trained — that is by construction, since the hybrid keeps
Faster R-CNN's backbone/RPN/regressor and replaces only the classification head. The mimic
metric is conditioned on that deployment distribution, which is precisely the distribution
that matters.

## 7. One-paragraph thesis framing

Replacing Faster R-CNN's MLP classification head with a TAO-trained sparse oblique decision
tree over raw RoI-Align features yields perfectly faithful, structurally interpretable
per-detection explanations. Two properties of hard decision trees initially degraded detection
metrics — piecewise-constant leaf scores collapse ranking-based metrics and Soft-NMS — and
mimic accuracy saturates at the teacher's own decision-boundary noise. Both are diagnosed and
addressed without sacrificing interpretability: routing-margin confidence turns the tree's own
hyperplane distances into a continuous, fully symbolic detection score, and confidence- plus
class-weighted TAO training focuses tree capacity on the teacher's reliable labels. The
resulting neuro-symbolic detector matches the black-box teacher (mAP@0.5:0.95 0.755 = 0.755)
and exceeds it on precision and F1, while its residual gap is fully attributable — and
empirically attributed — to irreducible teacher label noise.
