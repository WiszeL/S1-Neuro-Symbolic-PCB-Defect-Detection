# Neuro-Symbolic Printed Circuit Board Defect Detection with Faithful Per-Node Explanations

**[Author 1], [Author 2]**

[Department, University, City, Country]

email: [email of first author], [email of second author]

---

**ARTICLE INFO**

Article history: Received [dd Month yyyy] · Revised [dd Month yyyy] · Accepted [dd Month yyyy] · Available online [dd Month yyyy]

**Keywords:** Explainable artificial intelligence; Faithfulness; Neuro-symbolic; Printed circuit board; Sparse oblique decision tree

**IEEE style in citing this article:** [Authors], "Neuro-Symbolic Printed Circuit Board Defect Detection with Faithful Per-Node Explanations," *Journal of Innovation Information Technology and Application*, vol. [x], no. [x], pp. [x–x], 2026.

---

## ABSTRACT

Deep learning detectors inspect printed circuit boards (PCBs) with high accuracy, but their black-box decisions cannot be validated by technicians. Post-hoc explanation methods such as Grad-CAM only approximate the reasoning of the model and are therefore not guaranteed to be faithful. This study proposes a neuro-symbolic detector that replaces the classification head of a Faster R-CNN with a sparse oblique decision tree (SODT) trained by tree alternating optimization to mimic the labels of the network, with negative sampling and class weighting to balance the training. Detection scores are derived from the routing margins along the decision path, and every node on the path is explained by a heatmap obtained by projecting its sparse weights back through region-of-interest (RoI) alignment onto the feature pyramid, which yields an exact decomposition of the node decision. Evaluated on the six-class DeepPCB dataset, the neuro-symbolic model remains close to Faster R-CNN, with an mAP@0.5 of 0.982 against 0.987 and a precision of 0.912 against 0.903, and the routing margin raises mAP@0.5:0.95 from 0.644 to 0.787. Each explanation is compared with the random control of its own method: removing the patches ranked by the neuro-symbolic map changes the label of 96.1% of the detections against 59.1% for random patches, whereas Grad-CAM limited to the proposal reaches 68.7% against 60.3%. Detection with explanation takes 67.5 ms per image against 94.7 ms for Faster R-CNN with Grad-CAM. The per-node heatmaps let technicians trace each detection from the root to the leaf during validation.

---

## 1. INTRODUCTION

Printed circuit boards (PCBs) connect and mechanically support electronic components, so their quality determines the reliability of the whole electronic system. As PCB layouts become denser and more complex, defects caused by human error or machine faults during production become more likely [1], [2], and they degrade product performance and reliability, raise production costs, and can even pose safety risks [2], [3]. Conventional inspection, which relies on manual visual examination and electrical testing, is costly, time-consuming, and prone to human error [1], [2], [4]. These limitations have driven the adoption of automated inspection based on deep learning, which can detect surface defects in real time with high accuracy [5].

Object detectors, from one-stage models such as YOLO and SSD to two-stage models such as Faster R-CNN [6], form the basis of this automation [7] and have been widely applied to PCB defect detection [1]. Recent PCB detectors include Transformer-YOLO [8], YOLO-RLC [9], CDI-YOLO [3], PD-YOLOv8 [10], and YOLOv8-DEE [11], all reporting high accuracy on public PCB datasets. Within two-stage detection, Fung et al. [2] showed that a Faster R-CNN with a selective feature attention and pixel shuffle pyramid (SF-PSPyramid) neck improves the detection of very small PCB defects. Deep learning has therefore become a reliable basis for PCB inspection.

Despite this accuracy, these detectors are black boxes: they do not explain why a region is labeled as a particular defect. In high-stakes manufacturing such as PCB production, such explanations are needed so that technicians can validate the detection results and take responsibility for them [12], and without them, technicians' trust in the model decreases [13]. Explainable artificial intelligence (XAI) addresses this need by making the decision process of AI models traceable and understandable [14]. XAI has become the dominant approach to explainability in manufacturing inspection [12], and integrating it into visual quality assurance improves transparency, interpretability, and trust [4]. In PCB inspection, Tziolas et al. [13] used Deep SHAP and gradient-weighted class activation mapping (Grad-CAM) [15] to highlight the image areas that drive the decisions of a convolutional neural network (CNN).

These explanations, however, are post-hoc: they are produced after the black-box model has made its prediction and are not part of the decision itself. Post-hoc explanations only approximate the reasoning of the model and are therefore not guaranteed to be faithful, that is, to reflect the process that actually produced the prediction [16]–[18]. Saliency maps have also proven unreliable in high-stakes domains. In chest X-ray interpretation, seven saliency methods including Grad-CAM localized pathologies significantly worse than human experts, with the largest gap on small and complex structures [19], a characteristic shared by PCB defects. Technician validation built on unfaithful explanations may thus lead to wrong decisions, which makes faithfulness a prerequisite for useful explainability [16].

Neuro-symbolic (NeSy) architectures address this limitation by combining neural feature extraction with traceable symbolic reasoning, and because the symbolic component makes the decision itself, its explanation follows the actual decision process [20], [21]. A recent systematic review shows that NeSy methods are particularly promising for interpretability [22]. Among symbolic components, sparse oblique decision trees (SODTs) trained with tree alternating optimization (TAO) [23] can mimic a neural classifier with an accuracy close to the original while each node uses only a few features [24]. Kairgeldin and Carreira-Perpiñán [25] further composed CNN layers with an SODT into a NeSy classifier whose node weights can be mapped back to image regions. These studies address image classification, and to the best of our knowledge, how an SODT can serve as the decision component of an object detector, and how its decisions can be explained for individual detections, has not been studied.

This study proposes a NeSy PCB defect detector that integrates Faster R-CNN with an SODT, with three contributions. First, the SODT replaces the classification head of an SF-PSPyramid Faster R-CNN and is trained to mimic the detector labels on region proposals, with negative sampling balancing background against defects and class weighting balancing the defect classes against each other, while the regression head is retained. Second, a detection score based on routing margins restores the ranking that a single-label leaf cannot provide. Third, each node on the decision path is explained by a heatmap obtained by projecting its weights through RoI Align onto the feature pyramid, which yields an exact decomposition of the node decision, and these explanations are compared quantitatively with Grad-CAM on the DeepPCB dataset in terms of faithfulness, each against its own random control.

## 2. METHOD

The research proceeds in five stages: dataset preparation, training of the Faster R-CNN, extraction of a symbolic dataset from the trained detector, training of the SODT, and integration of both components into a single inference pipeline, followed by evaluation. The resulting architecture is shown in Figure 1.

![Figure 1](figures_bab4/gambar_3_4.png)

**Figure 1.** Architecture of the baseline Faster R-CNN (top) and the proposed neuro-symbolic model (bottom)

### 2.1. Dataset and Preprocessing

The DeepPCB dataset [26] contains 1,500 pairs of tested and template images with six defect classes: open, short, mousebite, spur, spurious copper, and pinhole. The official index files split it into 1,000 training images with 6,873 annotations and 500 test images with 3,140 annotations, and the largest class is only about 1.4 times the smallest in both subsets. This study is non-referential: only the tested image is used, because a template image is often unavailable in real inspection. The images of 640×640 pixels are normalized with ImageNet statistics. During training, they are randomly flipped horizontally and rescaled to a shorter side chosen from {480, 560, 640, 720, 800, 880} pixels, while inference, feature extraction, and evaluation use a fixed shorter side of 800 pixels, one of the training scales, so that the features passed to the symbolic component remain stable.

### 2.2. Neural Component: Faster R-CNN with SF-PSPyramid

The neural component follows the SF-PSPyramid Faster R-CNN of Fung et al. [2], [6]. A ResNet-50 backbone produces the feature maps C2–C5, and the SF-PSPyramid neck builds the high-resolution levels P2′ and P3′ from all backbone levels through pixel-shuffle blocks and selective feature attention instead of lateral connections [2], giving a pyramid P2′, P3′, P4, P5, and P6 with strides of 4, 8, 16, 32, and 64 pixels. The region proposal network (RPN) generates proposals, RoI Align pools each proposal from the pyramid level assigned by its size into a 7×7 grid, and a box head feeds a classification head and a regression head, followed by Soft-NMS [2]. The only modification of the architecture is the number of neck channels, reduced from 256 to 64 so that the input of the SODT is limited to 64×7×7 = 3,136 features. The anchor sizes of 32–512 pixels are chosen because [2] does not state them. The main configuration is summarized in Table 1.

**Table 1.** Main configuration of the neural and symbolic components

| Component | Parameter | Value |
|:--|:--|:--|
| Backbone and neck | Initialization; batch normalization | ResNet-50 pretrained on ImageNet; frozen |
| | Pyramid levels; channels | P2′, P3′, P4, P5, P6; 64 |
| RPN | Anchor sizes; aspect ratios | 32, 64, 128, 256, 512 pixels; 0.5, 1.0, 2.0 |
| | IoU thresholds (foreground / background) | ≥ 0.7 / < 0.3; best anchor rescued if IoU ≥ 0.3 |
| | Proposals after NMS (train/test) | 2,000 / 1,000 |
| RoI Align | Output size; sampling ratio | 7×7; 2 |
| Soft-NMS | Decay; sigma; IoU threshold; score cut | Linear; 0.5; 0.5; 0.001 |
| Detector training | Epochs; optimizer | 15; SGD (momentum 0.9, weight decay 0.0001) |
| | Learning rate; schedule | 0.02 with 500 warm-up iterations; ×0.1 at epochs 8 and 11 |
| | Gradient accumulation; precision | 4 steps; automatic mixed precision |
| SODT | Depth; initialization | 6; weights and biases from N(0, 1), random leaf labels |
| | TAO iterations; tolerance | 15; 10⁻⁶ |
| | λ; α; negative ratio | 0.01; 0.85; 2 |
| | Class weights ω | Short 2; spur 1.5; open 1.5; pinhole 1.25; spurious copper 1.25; mousebite and background 1 |
| Inference | Score threshold (Faster R-CNN / NeSy) | 0.5 / 0.35 |

RoI Align computes each output bin as an average of bilinearly interpolated feature values. For a given proposal, the interpolation coefficients depend only on the proposal geometry, so the output of channel *c* is a linear function of the feature map of that channel:

$$x_c = A\,F_c \tag{1}$$

where $x_c$ is the vector of 7×7 output values, $F_c$ is the feature map of channel *c* at the assigned pyramid level, and $A$ is the interpolation matrix shared by all channels. In addition, each feature map position corresponds to an image location given by the total stride of its level and to a receptive field around it [27], so values on the feature map can be traced back to image regions. Both properties are used in Section 2.5.

The detector follows the evaluation protocol of [2], i.e., AP computed in the PASCAL VOC manner and reported as AP50, AP75, AP80, AP85, and their mean AP@50:5:85, and is additionally evaluated with mAP@0.5:0.95 [7] and with precision, recall, and F1 at an IoU of 0.5 and a score of 0.5. The model with the best AP out of three trainings with different seeds is frozen and serves as the teacher.

### 2.3. Symbolic Component: Sparse Oblique Decision Tree

**Symbolic dataset.** The SODT is trained by model mimicking, using the predictions of the teacher as training labels [24]. For every image, all RPN proposals (up to 1,000) are kept rather than only the final detections, so that the SODT is trained on the same proposal population it will face at inference. The feature of each proposal is the RoI Align tensor of 64×7×7, taken before the box head and flattened to 3,136 values in a recorded order, so that every weight can be traced back to its channel and grid position. The label is the class with the highest softmax probability of the teacher, including background, not the ground truth, which is kept only for the detection metrics. This extraction yields 1,000,000 training RoIs and 500,000 test RoIs, about 82% of which are labeled as background.

**Tree model.** The SODT is a complete binary tree in which every internal node *i* applies a linear decision function to the feature vector $x$:

$$f_i(x) = w_i^{\top} x + b_i \tag{2}$$

and sends $x$ to the left child if $f_i(x) \ge 0$ and to the right child otherwise. Each leaf holds one class label, and the nodes from the root to the reached leaf form the decision path $P(x)$. Because $x$ is a flattened feature map, the weights $w_i$ can be reshaped into a 64×7×7 grid that shows which channels and positions the node uses [24], [25].

**Training.** The tree is trained to minimize a class-weighted loss with a sparsity penalty on the node weights:

$$E(\Theta) = \sum_{n=1}^{N} \omega_{y_n} L\big(y_n, T(x_n;\Theta)\big) + \lambda \sum_{i} h_\alpha(|\mathcal{R}_i|)\,\lVert w_i \rVert_1, \qquad h_\alpha(t) = t^{\alpha}\ (t > 0) \tag{3}$$

where $T(x_n;\Theta)$ is the tree prediction, $\omega_{y_n}$ is the weight of class $y_n$, $\mathcal{R}_i$ is the set of samples reaching node *i*, and $\lambda$ controls sparsity. With $\alpha > 0$, nodes that receive more samples are penalized more strongly, which counteracts the tendency of nodes near the root to become less sparse [25]. TAO [23], [24] optimizes the nodes alternately from the deepest level to the root. At each node, samples whose prediction does not depend on the direction taken are ignored, the remaining care set is labeled with the direction that yields a correct prediction, and this reduced binary problem is solved by L1-regularized logistic regression as a surrogate of the 0/1 loss, using a chunked L-BFGS-B solver in single precision because LIBLINEAR exhausted the 16 GB of memory available. An update is accepted only if it improves the reduced problem, so the objective never increases.

**Balanced mimic training.** Because the labels are teacher predictions and background dominates the proposals, the SODT is balanced at two levels. Negative sampling keeps all defect RoIs and a random subset of background RoIs at a negative ratio of 2, which gives 533,226 training RoIs (177,742 defect and 355,484 background), two thirds of them background, and balances background against defects. Class weighting follows cost-sensitive learning, which assigns higher misclassification costs to classes that would otherwise be neglected [28], and balances the defect classes against each other. It is applied at two places. In the reduced problem, each sample receives the weight

$$u_n = \lvert e_L(n) - e_R(n) \rvert \cdot \omega_{y_n} \tag{4}$$

where $e_L(n)$ and $e_R(n)$ are the 0/1 losses when sample *n* is sent left or right, so that only the care set contributes. At the leaves, the label is assigned by weighted majority:

$$\hat{y}_l = \arg\max_c\; \omega_c N_{l,c} \tag{5}$$

where $N_{l,c}$ is the number of samples of class *c* reaching leaf *l*. The weights of the classes with the weakest recall of the teacher are raised, whereas the background weight is kept at 1 so that false positives do not increase. Leaf labels are initialized randomly rather than by majority so that TAO does not collapse into the background class. After training, dead branches and pure subtrees are pruned without changing the decisions [23], [24], and $\lambda$ and $\alpha$ are chosen empirically to obtain a sparse tree without a noticeable loss of fidelity [24], [25]. Fidelity to the teacher is measured on the test RoIs with mimic accuracy, macro-F1, and per-class agreement, i.e., the fraction of RoIs of a teacher class that the SODT assigns to the same class.

### 2.4. Hybrid Inference with Routing Margin

The SODT replaces only the classification head, while the final box coordinates are still computed by the regression head of Faster R-CNN. Faithfulness therefore applies to the class decision, not to the box refinement. Replacing the head raises a scoring problem: every leaf stores one label, so all detections reaching leaves of the same class would receive an identical score, and both AP and Soft-NMS would lose their ranking. The detection score is therefore built from the margins of the nodes along the decision path:

$$s(x) = \prod_{i \in P(x),\; w_i \neq 0} \sigma\big(\lvert f_i(x) \rvert\big) \tag{6}$$

where $\sigma$ is the sigmoid function, and nodes whose weights were zeroed by pruning are skipped. Since each node function is the logit of an L1-regularized logistic regression [23], [25], every factor, which lies in [0.5, 1), represents the local confidence of the direction taken. The tree is still executed as a hard tree, so the decision path and label are unchanged, and the score depends only on the tree parameters. Because every factor is below 1, longer decision paths tend to receive lower scores.

At inference, Faster R-CNN produces the proposals and their RoI tensors, the SODT assigns the leaf label and the score of Equation (6), the regression head computes the boxes, detections labeled as background are removed, Soft-NMS is applied, and detections scored below the threshold are removed. The decision path, pyramid level, and neck feature map of each detection are stored for the heatmaps. A detection is displayed and counted in precision, recall, and F1 only if its score reaches the threshold. Faster R-CNN uses a threshold of 0.5, but the NeSy score is a product of sigmoids on a different scale from a softmax probability, so the same threshold is not comparable. The NeSy threshold is therefore set to the lowest multiple of 0.05 at which the precision on 200 random training images is at least that of Faster R-CNN at 0.5, which is 0.952. The NeSy model reaches 0.941 at 0.30 and 0.960 at 0.35, so the threshold is 0.35. It plays no role in training or in the decision path, and mAP does not depend on it.

### 2.5. Per-Node Heatmap

Every node on the decision path receives one heatmap that shows the region it weighs for the explained RoI. Both the node function in Equation (2) and RoI Align in Equation (1) are linear, so with the node weights of channel *c* reshaped to the 7×7 grid, $w_{i,c}$, the linear part of the node decision can be written directly in terms of the neck feature map:

$$f_i(x) - b_i = \sum_{c=1}^{C} w_{i,c}^{\top} A\, F_c \tag{7}$$

with $C = 64$. The gradient of Equation (7) with respect to $F_c$ is $A^{\top} w_{i,c}$, i.e., the node weights scattered back to the feature map positions they were pooled from. Multiplying this gradient by the feature values, as in Gradient × Input [18], gives the contribution of channel *c* at position *p*:

$$r_i[c,p] = d_i \cdot \big(A^{\top} w_{i,c}\big)_p \cdot F_c(p) \tag{8}$$

where $d_i \in \{+1, -1\}$ is the direction taken by node *i*, so a positive contribution supports that direction. Because every operation is linear, the contributions satisfy completeness [18] exactly:

$$\sum_{c=1}^{C} \sum_{p} r_i[c,p] = d_i \cdot \big(f_i(x) - b_i\big) \tag{9}$$

The bias is not attributed to any position, so the contributions sum to the linear part of the decision rather than to the full margin. For display, the channels are summarized into one map:

$$H_i(p) = \sum_{c=1}^{C} \lvert r_i[c,p] \rvert \tag{10}$$

The absolute value makes $H_i$ show the magnitude of influence, whether supporting or opposing, while the direction is shown by the decision path. Unlike the receptive-field density maps of [25], which depend only on the weights and are therefore the same for every input, $H_i$ uses the feature values of the RoI and is specific to each detection. It is also computed on the neck feature map rather than on the 7×7 grid. The computation is exact for each cell of the feature map at the pyramid level read by RoI Align, but one cell sees a much larger image region through the backbone [27], so the map is a label of the region being weighed, not a per-pixel cause, and it can only be nonzero inside the proposal. In practice, the weights are reshaped to 64×7×7 and multiplied by $d_i$, RoI Align is rerun on the neck feature map around the proposal, and one backward pass gives the gradients of Equation (8) for all nodes. The result is multiplied by the feature map and summed over channels, each cell is spread to the image pixels it covers according to the stride of its level, undoing scaling and padding, and the map is cropped to the proposal and normalized by its peak, as shown in Figure 2.

![Figure 2](figures_bab4/gambar_3_6.png)

**Figure 2.** Computation of the heatmap of one node on the decision path

### 2.6. Evaluation

**Detection.** The NeSy model is compared with Faster R-CNN on the 500 test images using the metrics of Section 2.2, each at its own score threshold (0.5 and 0.35), because precision, recall, and F1 depend on the threshold whereas mAP does not. Because the backbone, RPN, and regression head are shared, any difference comes from replacing the classification head. Two ablations isolate the contributions. The routing margin is tested by setting the score of every detection to 1 without changing its decision path or label. Class weighting is tested by training a tree without class weights with the same seed, hyperparameters, and negative ratio, whose threshold is derived with the same rule on the training images. The time per image is averaged over three runs for three pipelines: Faster R-CNN for detection only, NeSy for detection with per-node heatmaps, and Faster R-CNN with Grad-CAM maps.

**Explanation.** The explanations are evaluated with functionally-grounded metrics, i.e., quantitative measures that do not involve users [29], [30]. Grad-CAM [15], computed on the neck feature map, serves as the post-hoc baseline. Since Grad-CAM gives one map per detection while the SODT gives one map per node, the node maps are summed into a combined map for comparison only:

$$M(p) = \sum_{i \in P(x)} H_i(p) \tag{11}$$

Both methods are tested with the same protocol on the detections above their thresholds in the first 50 test images. The image is divided into patches of 16×16 pixels ranked by the map, the top patches, up to half of the number of patches touched by the proposal, form the selected set $S$, and each is replaced by the mean color of the image before the model is rerun and the label is checked. Necessity is the fraction of detections whose label changes when $S$ is removed, and Sufficiency is the fraction whose label is preserved when only $S$ is kept, corresponding to the deletion and preservation checks in [30]:

$$\text{Necessity} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\big[\hat{y}(x_n^{-S}) \neq \hat{y}(x_n)\big], \qquad \text{Sufficiency} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\big[\hat{y}(x_n^{S}) = \hat{y}(x_n)\big] \tag{12}$$

The Deletion and Insertion AUC [30] remove or add the patches incrementally in five steps and integrate the normalized score:

$$\text{AUC} = \sum_{k=1}^{K-1} \frac{s_k + s_{k+1}}{2}\,\Delta x_k \tag{13}$$

where $s_k$ is the normalized score at step *k* and $\Delta x_k$ is the fraction of patches modified between steps, so a low Deletion AUC and a high Insertion AUC indicate a faithful map. Following the recommendation to compare against random rankings [30], the procedure is repeated with a random patch order in which the patches inside the proposal come first. The two methods explain classification heads with different labels and scores, since the NeSy score is a routing margin whereas Grad-CAM uses a class probability, so only the comparison of each method with its own random control is meaningful, not the absolute values across methods. Grad-CAM is reported in two forms, on the whole image and with the patches restricted to the proposal, because the NeSy map can only be nonzero inside the proposal. Finally, the maps and the per-node heatmaps of two detections of different classes on one PCB are inspected qualitatively.

## 3. RESULTS AND DISCUSSION

### 3.1. Teacher Detector and SODT Fidelity

The detection performance of the teacher is summarized in Table 2, together with the SF-PSPyramid results reported by Fung et al. [2] in the same non-referential setting.

**Table 2.** Detection performance of the teacher Faster R-CNN on the DeepPCB test set (AP in %)

| Model | AP50 | AP75 | AP80 | AP85 | AP@50:5:85 |
|:--|:--|:--|:--|:--|:--|
| SF-PSPyramid [2] | 98.6 | **94.6** | **88.1** | **72.7** | **93.2** |
| Faster R-CNN (ours, 64 channels) | **98.73** | 94.30 | 87.50 | 71.63 | 93.03 |

Based on Table 2, the teacher is on par with [2]: AP50 is slightly higher, whereas AP at stricter IoU is 0.3–1.1 points lower, so AP@50:5:85 differs by only 0.17 points, although the neck is reduced to 64 channels. At an IoU of 0.5 and a score of 0.5, it reaches a precision of 0.903, a recall of 0.983, an F1 of 0.941, and an mAP@0.5:0.95 of 0.797. The box regression loss remained the largest loss component throughout training, so the main difficulty lies in the precision of the box boundary rather than in finding defects. Recall exceeds 0.96 for every class, whereas precision is lowest for short, pinhole, and spurious copper (0.827, 0.834, and 0.874). These errors come mainly from background regions detected as defects, 260 of the 331 false positives, which outnumber the 38 missed defects about seven times. Since the SODT mimics the teacher labels, it inherits this tendency to over-detect.

The structure and fidelity of the selected SODT are summarized in Table 3.

**Table 3.** Structure and fidelity of the selected SODT

| Metric | Value |
|:--|:--|
| Active internal nodes | 44 of 63 |
| Nonzero weights | 4,642 of 197,568 (sparsity 97.65%) |
| Average nonzero weights per active node | 105.5 (3.36% of 3,136 features) |
| Mimic accuracy (train / test) | 96.80% / 96.33% |
| Macro-F1 (test) | 0.909 |
| Agreement per class (test) | 0.951 (spurious copper) to 0.972 (spur) |

Based on Table 3, the SODT mimics the teacher with a fidelity above 96% although each node uses only about 3% of the features, in line with Hada et al. [24], and no class is left behind. During TAO, the mimic accuracy was already 94.9% at the first iteration and reached 96.8% at the third, whereas the number of nonzero weights fell from 9,715 to about 4,700, so TAO mainly simplified the tree rather than improving fidelity. Training stopped at iteration 8, and pruning reduced the 47 active nodes to the 44 in Table 3 without changing the decisions. The macro-F1 is lower than the mimic accuracy because it also accounts for precision, i.e., the background RoIs that are labeled as defects.

The selected tree after pruning is shown in Figure 3, where the number in each node is its count of nonzero weights, the letters at the leaves are the classes, and nodes are numbered level by level from the root N0.

![Figure 3](figures_bab4_v2/gambar_4_10.png)

**Figure 3.** Structure of the selected SODT after pruning (BG: background, OP: open, SH: short, MB: mousebite, SP: spur, SC: spurious copper, PH: pinhole)

Of the 64 leaves of the tree before pruning, 34 are labeled background and 30 are labeled with a defect, and in 9 of the 32 pairs of sibling leaves the two leaves hold different defect classes. The upper nodes therefore mainly filter background, whereas the nodes near the leaves decide the defect class, and the sparsity of every node keeps each of them compact enough to be turned into a heatmap in Section 3.5.

### 3.2. Balanced Mimic Training

Class weighting and the negative ratio are tested by retraining the tree with one change at a time, as summarized in Table 4.

**Table 4.** Effect of class weighting (CW) and negative ratio (NR) on the fidelity of the SODT on the test RoIs

| Tree | Mimic accuracy | Macro-F1 | Defect agreement | Background agreement | Class spread (points) |
|:--|:--|:--|:--|:--|:--|
| CW, NR 2 (selected) | 0.9633 | 0.9092 | 0.962 | 0.964 | **2.1** |
| No CW, NR 2 | **0.9667** | **0.9155** | – | – | 3.7 |
| CW, NR 1 | 0.9530 | 0.8887 | **0.976** | 0.948 | – |
| CW, NR full | 0.9714 | 0.9245 | 0.927 | **0.981** | – |

*Note.* Defect agreement is the mean agreement of the six defect classes, and class spread is the range of agreement over the classes.

Based on Table 4, class weighting shifts agreement to the weak classes. Without it, short has the lowest agreement (0.9331), and the spread across classes is 3.7 points. With it, short rises by 3.1 points to 0.9639, open by 1.5, spur by 1.3, pinhole by 0.6, and spurious copper by 0.5, whereas mousebite stays almost unchanged, so the spread narrows to 2.1 points. As the usual trade-off of cost-sensitive learning [28], background agreement falls by 0.6 points, so that mimic accuracy falls by 0.3 points and macro-F1 from 0.9155 to 0.9092.

The negative ratio balances background against defects. Mimic accuracy and macro-F1 increase with more background, but this does not reflect the defect classes, since background agreement rises from 0.948 to 0.981 whereas the mean defect agreement falls from 0.976 to 0.927, with short dropping from 0.978 to 0.928 and spurious copper from 0.970 to 0.900. At NR 1 the tree favors defects over background (difference +0.028), and with all background it favors background over defects (−0.054). NR 2 is the only nearly balanced point (−0.002), so it was selected. It also yields a sparse tree, with 4,642 nonzero weights against 7,463 for NR 1.

### 3.3. Detection Performance

The detection performance of both models and the two ablations is summarized in Table 5.

**Table 5.** Detection performance on the DeepPCB test set, each model at its own score threshold

| Model | Threshold | mAP@0.5:0.95 | mAP@0.5 | Precision | Recall | F1 |
|:--|:--|:--|:--|:--|:--|:--|
| Faster R-CNN (teacher) | 0.5 | **0.797** | **0.987** | 0.903 | **0.983** | 0.941 |
| NeSy | 0.35 | 0.787 | 0.982 | 0.912 | 0.978 | 0.944 |
| NeSy without CW | 0.30 | 0.787 | 0.983 | 0.912 | 0.978 | 0.944 |
| NeSy without routing margin | 0.35 | 0.644 | 0.810 | 0.750 | 0.988 | 0.852 |

Based on Table 5, replacing the classification head with the SODT changes mAP@0.5:0.95 by only 0.010 and mAP@0.5 by only 0.005, whereas precision rises by 0.9 points and F1 by 0.3 points, with recall falling by 0.5 points. These differences are small, so the NeSy model is read as on par with Faster R-CNN at each threshold, not better. Precision rises for five of the six classes, most for short (0.827 to 0.862) and spurious copper (0.874 to 0.891), which are among the weakest classes of Faster R-CNN, whereas it drops by 0.017 for spur. Recall falls by 0.002–0.007 and AP@0.5:0.95 by 0.001–0.018 for every class, because the SODT lowers the quality of the score ranking. The tree without class weighting gives the same mAP, precision, recall, and F1 to three decimals, so the gain in agreement of the weak classes in Section 3.2 does not appear at the detection level with the current weights, and the rise in precision cannot be attributed to class weighting.

The changes in precision and recall come from the types of error, as summarized in Table 6.

**Table 6.** False positives by type on the test set

| Model | Background | Wrong class | Mislocalized | Duplicate | Total |
|:--|:--|:--|:--|:--|:--|
| Faster R-CNN (threshold 0.5) | 260 | 33 | 35 | 3 | 331 |
| NeSy (threshold 0.35) | **215** | 49 | **27** | 7 | **298** |
| NeSy without routing margin (threshold 0.35) | 517 | 248 | 131 | 140 | 1,036 |

As shown in Table 6, of the 260 background false positives of Faster R-CNN, 65 disappear in the NeSy model and only 20 new ones appear, so the total falls from 331 to 298, whereas wrong-class errors rise from 33 to 49 because the SODT more often picks the wrong class for an ambiguous RoI. Missed defects increase from 38 to 47. Of the defects detected by Faster R-CNN, 15 are lost, 14 of them because the NeSy score is below 0.35 and one because of a wrong class. The weakest node on the decision path, i.e., the one with the smallest sigmoid factor, is the root in 9 of the 14 cases, so the NeSy model rejects detections whose first decision is uncertain, whether background or a faint defect.

The routing margin is essential, as shown by the last row of Table 5. Without it, all scores equal 1, the ordering needed by Soft-NMS and AP is lost, and mAP@0.5:0.95 falls by about 0.14 and precision by 16.2 points, whereas recall rises by 1.0 point. As shown in Table 6, false positives rise from 298 to 1,036, mainly because duplicates can no longer be separated by Soft-NMS (7 to 140) and background can no longer be filtered (215 to 517). The routing margin cannot change the label of a false positive, but it ranks it below the true positives, so the SODT can replace the classification head without losing detection ability. The bias of the score against long decision paths remains, and appears as the slightly lower AP of every class.

### 3.4. Faithfulness of Explanations

The faithfulness of the NeSy and Grad-CAM explanations, each with its own random control, is summarized in Table 7.

**Table 7.** Faithfulness of the explanations of NeSy and Grad-CAM against their own random controls

| Method | Necessity ↑ | Sufficiency ↑ | Deletion AUC ↓ | Insertion AUC ↑ | Detections |
|:--|:--|:--|:--|:--|:--|
| NeSy: map M | **0.961** | **0.774** | **0.439** | **0.758** | 337 |
| NeSy: random control | 0.591 | 0.285 | 0.590 | 0.585 | 337 |
| Grad-CAM, whole image | 0.322 | 0.183 | 0.711 | 0.186 | 345 |
| Grad-CAM, patches restricted to the proposal | 0.687 | 0.470 | 0.375 | 0.492 | 345 |
| Grad-CAM: random control | 0.603 | 0.301 | 0.432 | 0.402 | 345 |

As shown in Table 7, replacing the patches ranked by the NeSy map changes the label of 96.1% of the detections, against 59.1% for random patches, and keeping only these patches preserves the label of 77.4%, against 28.5%. The gap to the random control is about +0.37 for Necessity and +0.49 for Sufficiency, whereas for Grad-CAM restricted to the proposal it is only +0.08 and +0.17. The Deletion and Insertion AUC show the same direction, with gaps of 0.151 and 0.173 for NeSy against 0.057 and 0.090 for Grad-CAM. On the whole image, Grad-CAM is even below its random control, because most of its top patches lie outside the proposal, whereas the NeSy map is nonzero only inside it. The comparison is made only between each method and its own random control, because the two methods explain different classification heads. The regions highlighted by the SODT therefore determine the decision of the tree, in line with the faithful design of Section 2.5.

In terms of time, Faster R-CNN for detection only needs 54.6 ms per image, the NeSy model with per-node heatmaps 67.5 ms, and Faster R-CNN with Grad-CAM 94.7 ms, about 1.4 times the NeSy time. The SODT explanation is cheaper because one backward pass gives the maps of all nodes, whereas Grad-CAM needs a separate backward pass.

### 3.5. Case Study: Tracing a Detection Along the Decision Path

The explanations are read on a PCB with five defects from five different classes. Figure 4 shows the combined map M of the NeSy detections and the Grad-CAM map for the same proposals.

![Figure 4](figures_bab4_v2/fig_xai_overview_pcb1.png)

**Figure 4.** Explanations on one PCB: (a) map M (NeSy) and (b) Grad-CAM map on the same proposals

Based on Figure 4, panel (a), the map M is lit inside every box and concentrates on the structure of its own defect: the gap in the conductor, the notch in the edge of the conductor, the small protrusion, the hole, and the copper blob. Grad-CAM in panel (b) is also limited to the proposal, but its highlights are weaker and more scattered, and hardly lit for the copper blob and the gap. Map M comes from the decision of the tree itself, whereas Grad-CAM approximates the decision of Faster R-CNN. Both are read as indications of location, not as per-pixel causes.

Figure 5 decomposes the map M of one detection into the maps of the nodes on its decision path. The product of the sigmoid factors on the path equals the detection score (0.95 × 0.93 × 0.99 × 0.97 × 1.00 × 0.99 ≈ 0.84, the difference from 0.833 coming from the rounding of the factors).

![Figure 5](figures_bab4_v2/fig_xai_path_a.png)

**Figure 5.** Mousebite detection (score 0.833, path N0–N2–N6–N14–N29–N59): (a) map M, (b) Grad-CAM map, and (c) per-node heatmaps along the decision path

In Figure 5, the map M concentrates on the right side of the notch. Grad-CAM shares this peak as a vertical line but also lights the bottom edge of the proposal, which is not a peak of M. In panel (c), different nodes weigh different parts of the same defect: N0, N14, N29, and N59 concentrate on the right side of the notch, whereas N2 and N6 weigh the left and lower-left of the proposal. The smallest factors are at N0 (0.95) and N2 (0.93), whereas the nodes below are close to 1, so the confidence is determined mainly at the upper levels. A spur detection on the same PCB (score 0.801, path N0–N2–N5–N11–N24–N49) shares N0 and N2 with the mousebite, but N2 goes left for the spur and right for the mousebite, so the two detections separate at depth 3 (N5 instead of N6). Its smallest factor is again at the root (0.85), so its score is determined mostly by the confidence at the root, in line with Section 3.3, where the root is the weakest node in most lost detections. For this detection, the map M and Grad-CAM also point to different parts of the proposal, and only M can be traced to its nodes. A reader can thus follow the decision of one detection from the root to the leaf together with the region weighed at each step, whereas Grad-CAM gives only one map without a sequence of decisions.

## 4. CONCLUSION

This study integrated Faster R-CNN with a sparse oblique decision tree as a neuro-symbolic architecture for PCB defect detection on the DeepPCB dataset. The teacher Faster R-CNN is on par with Fung et al. [2], with an AP@50:5:85 of 93.03 against 93.2, and the SODT that replaces its classification head remains close to it, with an mAP@0.5 of 0.982 against 0.987 and an mAP@0.5:0.95 of 0.787 against 0.797. Precision rises from 0.903 to 0.912 and recall falls from 0.983 to 0.978 at the threshold of each model, so the model is on par with Faster R-CNN, not better. Three contributions support this result. First, balanced mimic training, in which the negative ratio balances background against defects and class weighting narrows the spread of agreement between defect classes from 3.7 to 2.1 points, although class weighting did not change detection performance at the current weights. Second, the routing margin restored the ranking of scores, raising mAP@0.5:0.95 from 0.644 to 0.787 and reducing false positives from 1,036 to 298. Third, the per-node heatmaps are faithful: against its own random control, the NeSy map changes the label of 96.1% of the detections against 59.1%, whereas Grad-CAM restricted to the proposal reaches 68.7% against 60.3%, and detection with explanation takes 67.5 ms against 94.7 ms for Faster R-CNN with Grad-CAM. Each detection can thus be traced step by step from the root to the leaf, which supports the accountability of the results during technician validation. The study has limitations: the main results come from one SODT and one training seed, the SODT only mimics the teacher labels, so defects it fails to mimic add false negatives, the score is biased against long decision paths, the neck was reduced to 64 channels, and the explanations were evaluated only quantitatively. Future work should test larger class weights and several seeds, train the SODT directly against the ground truth, use the full 256 neck channels at the cost of a heavier TAO training on 12,544 features, and conduct a user study with PCB inspection technicians.

## ACKNOWLEDGEMENTS

[To be completed by the authors.] This article is derived from the undergraduate thesis of the first author at the Informatics Study Program, Faculty of Information Technology and Data Science, Universitas Sebelas Maret.

## AUTHORSHIP STATEMENT ON THE USE OF GENERATIVE AI

During the preparation of this manuscript, the authors used Claude (Anthropic) for brainstorming the structure of the article and for drafting and adapting the manuscript in English from the first author's undergraduate thesis, as well as for language improvement. The research idea, methodology, experiments, results, and figures are the authors' own, and no figure was generated or altered by AI. The authors reviewed, verified, and edited all generated text and every reference, and take full responsibility for the content of this article.

## REFERENCES

[1] X. Chen, Y. Wu, X. He, and W. Ming, "A comprehensive review of deep learning-based PCB defect detection," *IEEE Access*, vol. 11, pp. 139017–139038, 2023, doi: 10.1109/ACCESS.2023.3339561.

[2] K. C. Fung, K.-W. Xue, C.-M. Lai, K.-H. Lin, and K.-M. Lam, "Improving PCB defect detection using selective feature attention and pixel shuffle pyramid," *Results Eng.*, vol. 21, Art. no. 101992, 2024, doi: 10.1016/j.rineng.2024.101992.

[3] G. Xiao, S. Hou, and H. Zhou, "PCB defect detection algorithm based on CDI-YOLO," *Sci. Rep.*, vol. 14, Art. no. 7351, 2024, doi: 10.1038/s41598-024-57491-3.

[4] R. Hoffmann and C. Reich, "A systematic literature review on artificial intelligence and explainable artificial intelligence for visual quality assurance in manufacturing," *Electronics*, vol. 12, no. 22, Art. no. 4572, 2023, doi: 10.3390/electronics12224572.

[5] Y. Liu, C. Zhang, and X. Dong, "A survey of real-time surface defect inspection methods based on deep learning," *Artif. Intell. Rev.*, vol. 56, no. 10, pp. 12131–12170, 2023, doi: 10.1007/s10462-023-10475-7.

[6] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 6, pp. 1137–1149, 2017, doi: 10.1109/TPAMI.2016.2577031.

[7] Z. Zou, K. Chen, Z. Shi, Y. Guo, and J. Ye, "Object detection in 20 years: A survey," *Proc. IEEE*, vol. 111, no. 3, pp. 257–276, 2023, doi: 10.1109/JPROC.2023.3238524.

[8] W. Chen, Z. Huang, Q. Mu, and Y. Sun, "PCB defect detection method based on Transformer-YOLO," *IEEE Access*, vol. 10, pp. 129480–129489, 2022, doi: 10.1109/ACCESS.2022.3228206.

[9] Y. Wang *et al.*, "YOLO-RLC: An advanced target-detection algorithm for surface defects of printed circuit boards based on YOLOv5," *Comput. Mater. Contin.*, vol. 80, no. 3, pp. 4973–4995, 2024, doi: 10.32604/cmc.2024.055839.

[10] L. Bai and W. H. Xu, "Improved printed circuit board defect detection scheme," *Sci. Rep.*, vol. 15, Art. no. 2389, 2025, doi: 10.1038/s41598-025-85245-2.

[11] F. Yi, A. S. A. Mohamed, M. H. M. Noor, F. Che Ani, and Z. E. Zolkefli, "YOLOv8-DEE: A high-precision model for printed circuit board defect detection," *PeerJ Comput. Sci.*, vol. 10, Art. no. e2548, 2024, doi: 10.7717/peerj-cs.2548.

[12] G. Tzionis *et al.*, "A review of explainable AI methods and their application in manufacturing systems," *Discov. Appl. Sci.*, vol. 8, no. 1, Art. no. 52, 2026, doi: 10.1007/s42452-025-07908-z.

[13] T. Tziolas *et al.*, "Explainable AI methods for identification of glue volume deficiencies in printed circuit boards," *Appl. Sci.*, vol. 15, no. 16, Art. no. 9061, 2025, doi: 10.3390/app15169061.

[14] S. Ali *et al.*, "Explainable artificial intelligence (XAI): What we know and what is left to attain trustworthy artificial intelligence," *Inf. Fusion*, vol. 99, Art. no. 101805, 2023, doi: 10.1016/j.inffus.2023.101805.

[15] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[16] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nat. Mach. Intell.*, vol. 1, no. 5, pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.

[17] C. Rudin, C. Chen, Z. Chen, H. Huang, L. Semenova, and C. Zhong, "Interpretable machine learning: Fundamental principles and 10 grand challenges," *Stat. Surv.*, vol. 16, pp. 1–85, 2022, doi: 10.1214/21-SS133.

[18] Q. Lyu, M. Apidianaki, and C. Callison-Burch, "Towards faithful model explanation in NLP: A survey," *Comput. Linguist.*, vol. 50, no. 2, pp. 657–723, 2024, doi: 10.1162/coli_a_00511.

[19] A. Saporta *et al.*, "Benchmarking saliency methods for chest X-ray interpretation," *Nat. Mach. Intell.*, vol. 4, no. 10, pp. 867–878, 2022, doi: 10.1038/s42256-022-00536-x.

[20] A. d'Avila Garcez and L. C. Lamb, "Neurosymbolic AI: The 3rd wave," *Artif. Intell. Rev.*, vol. 56, no. 11, pp. 12387–12406, 2023, doi: 10.1007/s10462-023-10448-w.

[21] H. A. Kautz, "The third AI summer: AAAI Robert S. Engelmore Memorial Lecture," *AI Mag.*, vol. 43, no. 1, pp. 105–125, 2022, doi: 10.1002/aaai.12036.

[22] C. Michel-Delétie and M. K. Sarker, "Neuro-symbolic methods for trustworthy AI: A systematic review with a focus on interpretability," *Neurosymbolic Artif. Intell.*, vol. 2, 2026, doi: 10.1177/29498732261469336.

[23] M. Á. Carreira-Perpiñán and P. Tavallali, "Alternating optimization of decision trees, with application to learning sparse oblique trees," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018, pp. 1211–1221.

[24] S. S. Hada, M. Á. Carreira-Perpiñán, and A. Zharmagambetov, "Sparse oblique decision trees: A tool to understand and manipulate neural net features," *Data Min. Knowl. Discov.*, vol. 38, no. 5, pp. 2863–2902, 2024, doi: 10.1007/s10618-022-00892-7.

[25] R. Kairgeldin and M. Á. Carreira-Perpiñán, "Neurosymbolic models based on hybrids of convolutional neural networks and decision trees," in *Proc. 19th Int. Conf. Neurosymbolic Learn. Reason. (NeSy)*, PMLR, vol. 284, 2025, pp. 796–813.

[26] S. Tang, F. He, X. Huang, and J. Yang, "Online PCB defect detector on a new PCB defect dataset," arXiv:1902.06197, 2019.

[27] X. Zhao, L. Wang, Y. Zhang, X. Han, M. Deveci, and M. Parmar, "A review of convolutional neural networks in computer vision," *Artif. Intell. Rev.*, vol. 57, no. 4, Art. no. 99, 2024, doi: 10.1007/s10462-024-10721-6.

[28] I. Araf, A. Idri, and I. Chairi, "Cost-sensitive learning for imbalanced medical data: A review," *Artif. Intell. Rev.*, vol. 57, no. 4, Art. no. 80, 2024, doi: 10.1007/s10462-023-10652-8.

[29] D. Canha, S. Kubler, K. Främling, and G. Fagherazzi, "A functionally-grounded benchmark framework for XAI methods: Insights and foundations from a systematic literature review," *ACM Comput. Surv.*, vol. 57, no. 12, Art. no. 320, 2025, doi: 10.1145/3737445.

[30] M. Nauta *et al.*, "From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI," *ACM Comput. Surv.*, vol. 55, no. 13s, Art. no. 295, 2023, doi: 10.1145/3583558.
