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

Deep learning detectors inspect printed circuit boards (PCBs) with high accuracy, but their black-box decisions cannot be validated by technicians. Post-hoc explanation methods such as Grad-CAM only approximate the reasoning of the model and are therefore not guaranteed to be faithful. This study proposes a neuro-symbolic detector that replaces the classification head of a Faster R-CNN with a sparse oblique decision tree (SODT) trained by tree alternating optimization to mimic the labels of the network. Detection scores are derived from the routing margins along the decision path, and every node on the path is explained by a heatmap obtained by projecting its sparse weights back through region-of-interest (RoI) alignment onto the feature pyramid, which yields an exact decomposition of the node decision. Evaluated on the six-class DeepPCB dataset, the neuro-symbolic model preserves the accuracy of Faster R-CNN, with an mAP@0.5 of 0.974 against 0.979 and a slightly higher precision. Its explanations are considerably more faithful than Grad-CAM, with a Necessity of 0.768 against 0.126 and a Deletion AUC of 0.200 against 0.599, while localization remains comparable and the full pipeline with explanations runs about 1.3 times faster than Faster R-CNN with Grad-CAM. Per-node heatmaps remain more faithful than random controls at every tree depth and reveal a consistent reasoning order, from the context around a defect to the defect contour, so that each detection can be traced step by step during technician validation.

---

## 1. INTRODUCTION

Printed circuit boards (PCBs) connect and mechanically support electronic components, so their quality determines the reliability of the whole electronic system. As PCB layouts become denser and more complex, defects caused by human error or machine faults during production become more likely [1], [2], and they degrade product performance and reliability, raise production costs, and can even pose safety risks [2], [3]. Conventional inspection, which relies on manual visual examination and electrical testing, is costly, time-consuming, and prone to human error [1], [2], [4]. These limitations have driven the adoption of automated inspection based on deep learning, which can detect surface defects in real time with high accuracy [5].

Object detectors, from one-stage models such as YOLO and SSD to two-stage models such as Faster R-CNN [6], form the basis of this automation [7] and have been widely applied to PCB defect detection [1]. Recent PCB detectors include Transformer-YOLO [8], YOLO-RLC [9], CDI-YOLO [3], PD-YOLOv8 [10], and YOLOv8-DEE [11], all reporting high accuracy on public PCB datasets. Within two-stage detection, Fung et al. [2] showed that a Faster R-CNN with a selective feature attention and pixel shuffle pyramid (SF-PSPyramid) neck improves the detection of very small PCB defects. Deep learning has therefore become a reliable basis for PCB inspection.

Despite this accuracy, these detectors are black boxes: they do not explain why a region is labeled as a particular defect. In high-stakes manufacturing such as PCB production, such explanations are needed so that technicians can validate the detection results and take responsibility for them [12], and without them, technicians' trust in the model decreases [13]. Explainable artificial intelligence (XAI) addresses this need by making the decision process of AI models traceable and understandable [14]. XAI has become the dominant approach to explainability in manufacturing inspection [12], and integrating it into visual quality assurance improves transparency, interpretability, and trust [4]. In PCB inspection, Tziolas et al. [13] used Deep SHAP and gradient-weighted class activation mapping (Grad-CAM) [15] to highlight the image areas that drive the decisions of a convolutional neural network (CNN).

These explanations, however, are post-hoc: they are produced after the black-box model has made its prediction and are not part of the decision itself. Post-hoc explanations only approximate the reasoning of the model and are therefore not guaranteed to be faithful, that is, to reflect the process that actually produced the prediction [16]–[18]. Saliency maps have also proven unreliable in high-stakes domains. In chest X-ray interpretation, seven saliency methods including Grad-CAM localized pathologies significantly worse than human experts, with the largest gap on small and complex structures [19], a characteristic shared by PCB defects. Technician validation built on unfaithful explanations may thus lead to wrong decisions, which makes faithfulness a prerequisite for useful explainability [16].

Neuro-symbolic (NeSy) architectures address this limitation by combining neural feature extraction with traceable symbolic reasoning, and because the symbolic component makes the decision itself, its explanation follows the actual decision process [20], [21]. A recent systematic review shows that NeSy methods are particularly promising for interpretability [22]. Among symbolic components, sparse oblique decision trees (SODTs) trained with tree alternating optimization (TAO) [23] can mimic a neural classifier with an accuracy close to the original while each node uses only a few features [24]. Kairgeldin and Carreira-Perpiñán [25] further composed CNN layers with an SODT into a NeSy classifier whose node weights can be mapped back to image regions. These studies address image classification, and to the best of our knowledge, how an SODT can serve as the decision component of an object detector, and how its decisions can be explained for individual detections, has not been studied.

This study proposes a NeSy PCB defect detector that integrates Faster R-CNN with an SODT, with four contributions. First, the SODT replaces the classification head of an SF-PSPyramid Faster R-CNN and is trained to mimic the detector labels on region proposals, while the regression head is retained. Second, a detection score based on routing margins restores the ranking that a single-label leaf cannot provide. Third, each node on the decision path is explained by a heatmap obtained by projecting its weights through RoI Align onto the feature pyramid, which yields an exact decomposition of the node decision. Fourth, the explanations are compared quantitatively with Grad-CAM on the DeepPCB dataset in terms of faithfulness and localization, each against its own random control.

## 2. METHOD

The research proceeds in five stages: dataset preparation, training of the Faster R-CNN, extraction of a symbolic dataset from the trained detector, training of the SODT, and integration of both components into a single inference pipeline, followed by evaluation. The resulting architecture is shown in Figure 1.

![Figure 1](figures_bab4/gambar_3_4.png)

**Figure 1.** Architecture of the baseline Faster R-CNN (top) and the proposed neuro-symbolic model (bottom)

### 2.1. Dataset and Preprocessing

The DeepPCB dataset [26] contains 1,500 pairs of tested and template images with six defect classes: open, short, mousebite, spur, spurious copper, and pinhole. The official index files split it into 1,000 training images with 6,873 annotations and 500 test images with 3,140 annotations, and the largest class is only about 1.4 times the smallest in both subsets. This study is non-referential: only the tested image is used, because a template image is often unavailable in real inspection. The images of 640×640 pixels are normalized with ImageNet statistics. During training, they are randomly flipped horizontally and rescaled to a shorter side chosen from {480, 560, 640, 720, 800, 880} pixels, while inference and feature extraction use a fixed shorter side of 640 pixels so that the features passed to the symbolic component remain stable.

### 2.2. Neural Component: Faster R-CNN with SF-PSPyramid

The neural component follows the SF-PSPyramid Faster R-CNN of Fung et al. [2], [6]. A ResNet-50 backbone produces the feature maps C2–C5, and the SF-PSPyramid neck builds the high-resolution levels P2′ and P3′ from all backbone levels through pixel-shuffle blocks and selective feature attention instead of lateral connections [2], giving a pyramid P2′, P3′, P4, P5, and P6 with strides of 4, 8, 16, 32, and 64 pixels. The region proposal network (RPN) generates proposals, RoI Align pools each proposal from the pyramid level assigned by its size into a 7×7 grid, and a box head feeds a classification head and a regression head, followed by Soft-NMS [2]. The only modification is the number of neck channels, reduced from 256 to 64 so that the input of the SODT is limited to 64×7×7 = 3,136 features. The main configuration is summarized in Table 1.

**Table 1.** Main configuration of the neural and symbolic components

| Component | Parameter | Value |
|:--|:--|:--|
| Backbone and neck | Initialization; batch normalization | ResNet-50 pretrained on ImageNet; frozen |
| | Pyramid levels; channels | P2′, P3′, P4, P5, P6; 64 |
| RPN | Anchor sizes; aspect ratios | 16, 32, 64, 128, 256 pixels; 0.5, 1.0, 2.0 |
| | Proposals after NMS (train/test) | 2,000 / 1,000 |
| RoI Align | Output size; sampling ratio | 7×7; 2 |
| Soft-NMS | Decay; IoU threshold; score threshold | Linear; 0.5; 0.001 |
| Detector training | Epochs; optimizer | 15; SGD (momentum 0.9, weight decay 0.0001) |
| | Learning rate; schedule | 0.02 with 500 warm-up iterations; ×0.1 at epochs 8 and 11 |
| | Gradient accumulation; precision | 4 steps; automatic mixed precision |
| SODT | Depth; initialization | 6; weights and biases from N(0, 1), random leaf labels |
| | TAO iterations; tolerance | 15; 10⁻⁶ |
| | λ; α; negative ratio | 20; 0.15; 2 |
| | Class weights ω | Short 2; spur 1.5; open 1.5; pinhole 1.25; spurious copper 1.25; mousebite and background 1 |

RoI Align computes each output bin as an average of bilinearly interpolated feature values. For a given proposal, the interpolation coefficients depend only on the proposal geometry, so the output of channel *c* is a linear function of the feature map of that channel:

$$x_c = A\,F_c \tag{1}$$

where $x_c$ is the vector of 7×7 output values, $F_c$ is the feature map of channel *c* at the assigned pyramid level, and $A$ is the interpolation matrix shared by all channels. In addition, each feature map position corresponds to an image location given by the total stride of its level and to a receptive field around it [27], so values on the feature map can be traced back to image regions. Both properties are used in Section 2.5.

The detector is evaluated on the test set with mAP@0.5, mAP@0.75, mAP@0.5:0.85 for comparison with [2], and mAP@0.5:0.95 [7], together with precision, recall, and F1 at an IoU of 0.5 and a score of 0.5. The trained detector is then frozen and serves as the teacher.

### 2.3. Symbolic Component: Sparse Oblique Decision Tree

**Symbolic dataset.** The SODT is trained by model mimicking, using the predictions of the teacher as training labels [24]. For every image, all RPN proposals (up to 1,000) are kept rather than only the final detections, so that the SODT is trained on the same proposal population it will face at inference. The feature of each proposal is the RoI Align tensor of 64×7×7, taken before the box head and flattened to 3,136 values in a recorded order, so that every weight can be traced back to its channel and grid position. The label is the class with the highest softmax probability of the teacher, including background, not the ground truth, which is kept only for the localization metrics. This extraction yields 1,000,000 training RoIs and 500,000 test RoIs, more than 85% of which are labeled as background.

**Tree model.** The SODT is a complete binary tree in which every internal node *i* applies a linear decision function to the feature vector $x$:

$$f_i(x) = w_i^{\top} x + b_i \tag{2}$$

and sends $x$ to the left child if $f_i(x) \ge 0$ and to the right child otherwise. Each leaf holds one class label, and the nodes from the root to the reached leaf form the decision path $P(x)$. Because $x$ is a flattened feature map, the weights $w_i$ can be reshaped into a 64×7×7 grid that shows which channels and positions the node uses [24], [25].

**Training.** The tree is trained to minimize a class-weighted loss with a sparsity penalty on the node weights:

$$E(\Theta) = \sum_{n=1}^{N} \omega_{y_n} L\big(y_n, T(x_n;\Theta)\big) + \lambda \sum_{i} h_\alpha(|\mathcal{R}_i|)\,\lVert w_i \rVert_1, \qquad h_\alpha(t) = t^{\alpha}\ (t > 0) \tag{3}$$

where $T(x_n;\Theta)$ is the tree prediction, $\omega_{y_n}$ is the weight of class $y_n$, $\mathcal{R}_i$ is the set of samples reaching node *i*, and $\lambda$ controls sparsity. With $\alpha > 0$, nodes that receive more samples are penalized more strongly, which counteracts the tendency of nodes near the root to become less sparse [25]. TAO [23], [24] optimizes the nodes alternately from the deepest level to the root. At each node, samples whose prediction does not depend on the direction taken are ignored, the remaining care set is labeled with the direction that yields a correct prediction, and this reduced binary problem is solved by L1-regularized logistic regression as a surrogate of the 0/1 loss. An update is accepted only if it improves the reduced problem, so the objective never increases.

Class weighting follows cost-sensitive learning, which assigns higher misclassification costs to classes that would otherwise be neglected [28], and is applied at two places. In the reduced problem, each sample receives the weight

$$u_n = \lvert e_L(n) - e_R(n) \rvert \cdot \omega_{y_n} \tag{4}$$

where $e_L(n)$ and $e_R(n)$ are the 0/1 losses when sample *n* is sent left or right, so that only the care set contributes. At the leaves, the label is assigned by weighted majority:

$$\hat{y}_l = \arg\max_c\; \omega_c N_{l,c} \tag{5}$$

where $N_{l,c}$ is the number of samples of class *c* reaching leaf *l*. The background weight is kept at 1 so that false positives do not increase. Because background dominates the proposals, all defect RoIs are kept and background RoIs are sampled at a negative ratio of 2, giving 437,508 training RoIs of which 66.7% are background. Leaf labels are initialized randomly rather than by majority so that TAO does not collapse into the background class. After training, dead branches and pure subtrees are pruned without changing the decisions [23], [24], and $\lambda$ and $\alpha$ are chosen empirically to obtain a sparse tree without a noticeable loss of fidelity [24], [25]. Fidelity to the teacher is measured on the test RoIs with mimic accuracy, macro-F1, and per-class agreement, i.e., the fraction of RoIs of a teacher class that the SODT assigns to the same class.

### 2.4. Hybrid Inference with Routing Margin

The SODT replaces only the classification head, while the final box coordinates are still computed by the regression head of Faster R-CNN. Faithfulness therefore applies to the class decision, not to the box refinement. Replacing the head raises a scoring problem: every leaf stores one label, so all detections reaching leaves of the same class would receive an identical score, and both AP and Soft-NMS would lose their ranking. The detection score is therefore built from the margins of the nodes along the decision path:

$$s(x) = \prod_{i \in P(x),\; w_i \neq 0} \sigma\big(\lvert f_i(x) \rvert\big) \tag{6}$$

where $\sigma$ is the sigmoid function, and nodes whose weights were zeroed by pruning are skipped. Since each node function is the logit of an L1-regularized logistic regression [23], [25], every factor, which lies in [0.5, 1), represents the local confidence of the direction taken. The tree is still executed as a hard tree, so the decision path and label are unchanged, and the score depends only on the tree parameters. Because every factor is below 1, longer decision paths tend to receive lower scores. At inference, Faster R-CNN produces the proposals and their RoI tensors, the SODT assigns the leaf label and the score of Equation (6), the regression head computes the boxes, detections labeled as background or scored below the threshold are removed, and Soft-NMS is applied. The decision path, pyramid level, and neck feature map of each detection are stored for the heatmaps.

### 2.5. Per-Node Heatmap

Every node on the decision path receives one heatmap that shows the region it weighs for the explained RoI. Both the node function in Equation (2) and RoI Align in Equation (1) are linear, so with the node weights of channel *c* reshaped to the 7×7 grid, $w_{i,c}$, the linear part of the node decision can be written directly in terms of the neck feature map:

$$f_i(x) - b_i = \sum_{c=1}^{C} w_{i,c}^{\top} A\, F_c \tag{7}$$

with $C = 64$. The gradient of Equation (7) with respect to $F_c$ is $A^{\top} w_{i,c}$, i.e., the node weights scattered back to the feature map positions they were pooled from. Multiplying this gradient by the feature values, as in Gradient × Input [18], gives the contribution of channel *c* at position *p*:

$$r_i[c,p] = d_i \cdot \big(A^{\top} w_{i,c}\big)_p \cdot F_c(p) \tag{8}$$

where $d_i \in \{+1, -1\}$ is the direction taken by node *i*, so a positive contribution supports that direction. Because every operation is linear, the contributions satisfy completeness [18] exactly:

$$\sum_{c=1}^{C} \sum_{p} r_i[c,p] = d_i \cdot \big(f_i(x) - b_i\big) \tag{9}$$

The bias is not attributed to any position, so the contributions sum to the linear part of the decision rather than to the full margin. For display, the channels are summarized into one map:

$$H_i(p) = \sum_{c=1}^{C} \lvert r_i[c,p] \rvert \tag{10}$$

The absolute value makes $H_i$ show the magnitude of influence, whether supporting or opposing, while the direction is shown by the decision path. Unlike the receptive-field density maps of [25], which depend only on the weights and are therefore the same for every input, $H_i$ uses the feature values of the RoI and is specific to each detection. It is also computed on the neck feature map, hereafter called the FPN projection, rather than on the 7×7 grid. Its resolution is therefore finer than the grid, but it remains at the region level because the exact computation stops at the neck, where each position covers a receptive field in the image [27]. In practice, the weights are reshaped to 64×7×7 and multiplied by $d_i$, RoI Align is rerun on the neck feature map and back-propagated to obtain Equation (8), the result is summed over channels, cropped to the proposal using the stride of its pyramid level, normalized, and upsampled bilinearly to the proposal size, as shown in Figure 2.

![Figure 2](figures_bab4/gambar_3_6.png)

**Figure 2.** Computation of the heatmap of one node on the decision path

### 2.6. Evaluation

**Detection.** The NeSy model is compared with Faster R-CNN on the 500 test images using the metrics of Section 2.2. Because the backbone, RPN, and regression head are shared, any difference comes from replacing the classification head. The time per image is averaged over three runs for three pipelines: Faster R-CNN for detection only, NeSy for detection with per-node heatmaps, and Faster R-CNN with Grad-CAM maps, where map generation is timed up to the normalized map.

**Explanation.** The explanations are evaluated with functionally-grounded metrics, i.e., quantitative measures that do not involve users [29], [30]. Grad-CAM [15], computed on the neck feature map as the last convolutional layer before the classification head, serves as the post-hoc baseline. Since Grad-CAM gives one map per detection while the SODT gives one map per node, the node maps are summed into a combined map for comparison only:

$$M(p) = \sum_{i \in P(x)} H_i(p) \tag{11}$$

All detections with a score of at least 0.5 are evaluated. For path-level faithfulness, the top 50% positions of each map inside the proposal, extended by two positions, are selected on the neck feature map, RoI Align is rerun, and the label is checked. Necessity is the fraction of detections whose label changes when the selected set $S$ is zeroed, and Sufficiency is the fraction whose label is preserved when only $S$ is kept, corresponding to the deletion and preservation checks in [30]:

$$\text{Necessity} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\big[\hat{y}(x_n^{-S}) \neq \hat{y}(x_n)\big], \qquad \text{Sufficiency} = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}\big[\hat{y}(x_n^{S}) = \hat{y}(x_n)\big] \tag{12}$$

The Deletion and Insertion AUC [30] remove or add the selected positions incrementally in five steps and integrate the normalized score:

$$\text{AUC} = \sum_{k=1}^{K-1} \frac{s_k + s_{k+1}}{2}\,\Delta x_k \tag{13}$$

where $s_k$ is the normalized score at step *k* and $\Delta x_k$ is the fraction of positions modified between steps, so a low Deletion AUC and a high Insertion AUC indicate a faithful map. Following the recommendation to compare against random rankings [30], the same procedure is repeated with randomly selected positions. Because the NeSy score is a routing margin whereas Grad-CAM uses a class probability, each method is compared with its own random control. Localization is measured with the Pointing Game, i.e., whether the maximum of the map falls inside the ground-truth box, and with the heatmap IoU between the binarized map and the box [30], after resampling each map to 7×7. These metrics are computed on low-IoU proposals, i.e., proposals with an IoU of 0.05–0.35 to the ground truth that both models label as a defect, because a well-aligned proposal would make pointing trivial. Per-node faithfulness removes the top 50% of each $H_i$, checks whether the sign of $f_i(x)$ flips (Necessity), and computes the Deletion and Insertion AUC on the routing margin of the node. Finally, two randomly chosen true positives per class are inspected qualitatively.

## 3. RESULTS AND DISCUSSION

### 3.1. Teacher Detector and SODT Fidelity

The detection performance of both models is summarized in Table 2, together with the SF-PSPyramid results reported by Fung et al. [2] in the same non-referential setting.

**Table 2.** Detection performance and inference time on the DeepPCB test set

| Model | mAP@0.5 | mAP@0.75 | mAP@0.5:0.85 | mAP@0.5:0.95 | Precision | Recall | F1 | Time (ms/image) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| SF-PSPyramid [2] | 0.986 | 0.946 | 0.932 | – | – | – | – | – |
| Faster R-CNN (ours, 64 channels) | 0.979 | 0.920 | 0.901 | 0.759 | 0.910 | 0.982 | 0.945 | 76.9 ± 30.9 |
| NeSy (ours) | 0.974 | – | – | 0.757 | 0.923 | 0.973 | 0.947 | 80.9 ± 11.7 |

As shown in Table 2, the teacher detects almost all defects, but its mAP drops by about 0.22 from mAP@0.5 to mAP@0.5:0.95, in line with the box regression loss that remained the largest loss component throughout training. Its mAP@0.5:0.85 is 0.031 below SF-PSPyramid, consistent with the neck reduced to 64 channels. Its main errors are background regions detected as defects, especially pinhole and short with precisions of 0.799 and 0.859, and these false positives outnumber missed defects about six times. Since the SODT mimics the teacher labels, it inherits this tendency to over-detect.

The structure and fidelity of the selected SODT are summarized in Table 3.

**Table 3.** Structure and fidelity of the selected SODT

| Metric | Value |
|:--|:--|
| Active internal nodes | 32 of 63 |
| Nonzero weights | 4,057 of 197,568 (sparsity 97.9%) |
| Average nonzero weights per active node | 126.8 (4.0% of 3,136 features) |
| Mimic accuracy (train / test) | 96.51% / 96.26% |
| Macro-F1 (test) | 0.889 |

Based on Table 3, the SODT mimics the teacher with a fidelity above 96% although each node uses only about 4% of the features, in line with Hada et al. [24]. During TAO, the mimic accuracy exceeded 95% from the first iteration and rose by only about 1.5 points, while the number of nonzero weights was nearly halved, so TAO mainly simplified the tree rather than improving fidelity. The macro-F1 is much lower because about 16,400 background RoIs are labeled as defects, more than the RoIs of any single defect class. The negative ratio of 2 with class weighting was selected because it gave the most balanced per-class agreement, with a range of 1.4 points, against 3.6 points without class weighting and 4.0–8.5 points for negative ratios of 1, 4, and no sampling.

The pruned tree is shown in Figure 3, where the number in each node is its count of nonzero weights and nodes are numbered level by level from the root N0.

![Figure 3](figures_bab4/gambar_4_10.png)

**Figure 3.** Structure of the selected SODT after pruning

As shown in Figure 3, background exits are spread over depths 3–6, whereas all defect leaves lie at depth 6, and 10 of the 13 bottom leaf pairs hold two different defect classes. The upper and middle nodes therefore mainly filter background, while the bottom nodes decide the defect class. With α = 0.15, the root and the nodes on the main path keep 249–534 nonzero weights, whereas nodes at depth 5 keep only 8–131. The subtrees under N19 and N22, however, contain only defect leaves, so a background RoI that enters them is necessarily labeled as a defect, which is a source of false positives beyond those of the teacher.

### 3.2. Neuro-Symbolic Detection Performance

Based on Table 2, replacing the classification head with the SODT changes the detection performance only marginally: precision rises slightly and recall falls slightly, so the fidelity of the SODT carries over to the detection level. The mAP@0.5 of 0.974 is also within about 0.01 of the values reported for recent detectors on DeepPCB [2], [11], although the evaluation protocols differ. The NeSy model is only slightly slower than Faster R-CNN without explanation, despite generating per-node heatmaps, because its detection stage is lighter. Each RoI in the NeSy model yields a single class candidate and RoIs ending at a background leaf do not pass the score threshold, whereas Faster R-CNN forwards nearly every class of each RoI to the sequential Soft-NMS, which also explains its larger variance.

Comparing the confusion matrices of both models traces these changes to their sources. The number of missed defects rises from 46 to 65, whereas background false positives fall from 292 to 236. The additional misses come from defect RoIs that the SODT fails to mimic because its training is dominated by background. At the class level, misses do not increase for short, the class with the highest weight and agreement, nor for open, but they increase for the other four classes, most for spur, which has the lowest agreement. Background false positives increase only for short, whose precision drops from 0.859 to 0.817 because class weighting makes the SODT more willing to assign the class with the largest weight, whereas the precision of spurious copper and pinhole rises to 0.983 and 0.858. The tendencies of the SODT, leaning toward background and balanced by class weighting on short, thus propagate to the detection level.

### 3.3. Effect of the Routing Margin

The role of the routing margin is tested by setting the score of every detection to 1 without changing its decision path or label, as summarized in Table 4.

**Table 4.** Effect of the routing margin on NeSy detection

| Metric | With routing margin | Without routing margin | Δ |
|:--|:--|:--|:--|
| mAP@0.5 | 0.974 | 0.821 | +0.153 |
| Precision | 0.923 | 0.787 | +0.135 |
| Recall | 0.973 | 0.983 | −0.010 |
| F1 | 0.947 | 0.874 | +0.073 |
| AP@0.5 short | 0.950 | 0.706 | +0.244 |
| AP@0.5 pinhole | 0.993 | 0.801 | +0.192 |

As shown in Table 4, removing the routing margin sharply reduces mAP and precision while recall rises slightly. Background false positives increase about 3.5 times, from 236 to 817, and confusions between defect classes from 27 to 127. With uniform scores, low-confidence detections are no longer filtered by the score threshold, and AP loses the ordering between high- and low-confidence detections. The largest AP drops occur for short and pinhole, the two classes with the most background false positives, so the background leakage of the SODT observed in Section 3.1 is suppressed mainly by the routing margin. In a pair of pinhole detections, for example, a false positive lies close to the split at N4, N9, and N19 before entering the N19 subtree, whereas a true positive stays far from every split. The routing margin cannot correct the label of the false positive, but it ranks it below the true positive.

### 3.4. Faithfulness and Localization of Explanations

The path-level faithfulness and localization of the NeSy explanations and Grad-CAM are summarized in Table 5, with the random control of each method.

**Table 5.** Path-level faithfulness and localization of NeSy and Grad-CAM explanations

| Method | Necessity ↑ | Sufficiency ↑ | Deletion AUC ↓ | Insertion AUC ↑ | Pointing Game ↑ | Heatmap IoU ↑ |
|:--|:--|:--|:--|:--|:--|:--|
| NeSy | 0.768 | 1.000 | 0.200 | 0.877 | 0.863 | 0.769 |
| NeSy random | 0.022 | 0.981 | 0.592 | 0.590 | – | – |
| Grad-CAM | 0.126 | 0.975 | 0.599 | 0.846 | 0.882 | 0.790 |
| Grad-CAM random | 0.018 | 0.983 | 0.758 | 0.758 | – | – |
| Random map | – | – | – | – | 0.820 | 0.760 |

Based on Table 5, the NeSy map is far from its random control on Necessity, Deletion AUC, and Insertion AUC, and its Necessity is about six times that of Grad-CAM, whereas Grad-CAM is only slightly better than its own random control. The NeSy map is an exact decomposition of the contribution of each node, following Equations (8)–(10), whereas Grad-CAM averages the gradients of each channel and discards the negative part through its ReLU, so removing the regions it highlights rarely changes the label. Sufficiency is almost complete for every row, including the random controls, so this metric does not separate the two methods. For localization, NeSy and Grad-CAM are nearly equal and both are above the random map, so the NeSy heatmaps point to defect regions as well as the common post-hoc method. The margin over the random map is small because one position on the neck feature map covers an image region much larger than a low-IoU proposal. With comparable localization, the two methods differ in faithfulness. Grad-CAM also needs 102.5 ± 32.5 ms per image, about 1.3 times the NeSy time in Table 2, because the NeSy model only reruns RoI Align on feature maps that are already available, whereas Grad-CAM reruns the backbone and back-propagates to the neck for every detection.

The combined map is used only for this comparison, since the actual NeSy explanation is the heatmap of each node. Its faithfulness is evaluated on about 3,310 detections at each depth, as shown in Table 6, where whole-box Necessity is the Necessity obtained when the entire proposal is removed and support is the fraction of the proposal that receives a contribution from the node.

**Table 6.** Per-node faithfulness at each depth of the decision path

| Depth | Necessity (NeSy / random) | Whole-box Necessity | Deletion AUC (NeSy / random) | Insertion AUC (NeSy / random) | Support |
|:--|:--|:--|:--|:--|:--|
| 1 | 0.057 / 0.000 | 0.000 | 0.611 / 0.859 | 0.947 / 0.859 | 0.544 |
| 2 | 0.298 / 0.001 | 0.999 | 0.641 / 0.889 | 0.948 / 0.889 | 0.544 |
| 3 | 0.364 / 0.006 | 0.824 | 0.624 / 0.859 | 0.944 / 0.858 | 0.544 |
| 4 | 0.336 / 0.008 | 0.623 | 0.629 / 0.886 | 0.949 / 0.885 | 0.544 |
| 5 | 0.353 / 0.002 | 0.347 | 0.566 / 0.877 | 0.950 / 0.877 | 0.416 |
| 6 | 0.386 / 0.011 | 0.297 | 0.570 / 0.855 | 0.951 / 0.856 | 0.297 |

As shown in Table 6, the map of each node is more faithful than random positions at every depth, and the random control almost never flips a node decision. The low Necessity at depth 1 does not indicate a wrong map, because the decision of N0 does not flip even when the whole proposal is removed. At depths 5 and 6, Necessity even exceeds whole-box Necessity, because deleting the highlighted positions removes the supporting evidence while the opposing evidence remains, and the shrinking support shows that deeper nodes weigh smaller regions.

### 3.5. Per-Node Heatmap Analysis

The difference between NeSy and Grad-CAM explanations is illustrated in Figure 4 for a spur detection, with the node heatmaps ordered along the decision path.

![Figure 4](figures_bab4/gambar_4_14.png)

**Figure 4.** Per-node NeSy heatmaps (top) and the Grad-CAM map (bottom) for one spur detection

As shown in Figure 4, Grad-CAM gives a single map that spreads beyond the proposal and along the copper trace, whereas the NeSy model shows the region weighed at each decision step. The heatmaps of two spur detections are shown in Figure 5.

![Figure 5](figures_bab4/gambar_4_15.png)

**Figure 5.** Per-node heatmaps of two spur detections along the decision path N0–N1–N4–N10–N22–N45

Based on Figure 5, both spur detections follow the same decision path. The tree first weighs the copper in the proposal corner and the band along its bottom edge at N0 and N1, then moves to the body of the protrusion at N4 and N10, and at N22 and N45 narrows to one side of the protrusion while the surrounding copper and substrate are no longer weighed. The same analysis for all six classes is summarized in Table 7.

**Table 7.** Regions weighed by the nodes for each defect class

| Class (decision path) | Depths 1–2 | Depths 3–4 | Depths 5–6 |
|:--|:--|:--|:--|
| Spur (N0–N1–N4–N10–N22–N45) | Copper at the corner and bottom edge of the proposal | Body of the protrusion | One side of the protrusion |
| Spurious copper (N0–N1–N4–N10–N22–N45) | Substrate at the corner and bottom edge of the proposal | Bottom edge of the copper blob, then the surrounding substrate | Edge of the blob |
| Pinhole (N0–N1–N4–N10–N22–N46) | Intact copper at the corner and bottom edge of the proposal | The hole, then the surrounding copper | Rim of the hole |
| Mousebite (N0–N1–N4–N10–N22–N46) | Proposal corner and the bitten trace edge | The bite indentation | Contour of the indentation |
| Open (N0–N1–N4–N10–N21–N43) | Around the gap and the bottom edge of the proposal | Broken trace ends, then the gap | Trace ends at the gap |
| Short (N0–N1–N4–N9–N19–N39) | Corner and bottom edge of the proposal | Copper bridge, then the bottom edge | Copper band connecting two traces |

Based on Table 7, two detections of the same class always follow the same decision path, and the order of weighing repeats across classes. Depths 1 and 2 weigh the context around the defect, mainly the corner and bottom edge of the proposal, with similar patterns for all classes. From depth 3 the weighing moves to the defect itself, and at depths 5 and 6 it narrows to the edge or contour of the defect rather than its center, consistent with the decreasing support in Table 6. Each decision step of the NeSy model can therefore be read as a specific region being weighed, and Table 6 shows that each of these maps is faithful to its node decision.

### 3.6. Effect of the FPN Projection

The FPN projection of Section 2.5 is tested by computing the heatmaps directly on the 7×7 RoI Align grid with the same tree and weights, while deletion is still performed on the neck feature map. The results are summarized in Table 8.

**Table 8.** Effect of the FPN projection on faithfulness and localization

| Metric | With FPN projection | Without FPN projection | Random map |
|:--|:--|:--|:--|
| Path-level Necessity | 0.768 | 0.712 | – |
| Path-level Deletion AUC | 0.200 | 0.222 | – |
| Path-level Insertion AUC | 0.877 | 0.846 | – |
| Pointing Game | 0.863 | 0.777 | 0.820 |
| Heatmap IoU | 0.769 | 0.757 | 0.760 |
| Node Necessity, depths 1–6 | 0.057, 0.298, 0.364, 0.336, 0.353, 0.386 | 0.016, 0.065, 0.206, 0.235, 0.153, 0.276 | – |

As shown in Table 8, without the FPN projection localization falls to the level of the random map, with the Pointing Game even below it, whereas path-level faithfulness drops only slightly. Node Necessity decreases at every depth, most sharply at depths 2 and 5. On the 7×7 grid, each cell represents a large part of the proposal and the location of the contribution within the cell is lost, so the map cannot point to the precise location. Faithfulness stays above the random control because the tree weights are unchanged, so what the projection preserves is the location of the map rather than the evidence being weighed. A mousebite detection with and without the projection is shown in Figure 6.

![Figure 6](figures_bab4/gambar_4_21.png)

**Figure 6.** Per-node heatmaps of a mousebite detection without (top) and with (bottom) the FPN projection

Based on Figure 6, without the projection N0 loses the weighing at the base of the bite and only the proposal corner remains, and the map of N4 shrinks to a single point at the base of the bite, whereas with the projection it covers the whole indentation. N22 spreads to the substrate left of the bite and to the right edge of the proposal, whereas with the projection it concentrates on the base of the bite. N1, N10, and N46 point to almost the same regions in both cases. This pattern agrees with the lower node Necessity in Table 8 and confirms that the FPN projection is needed for each node map to point to the correct location.

## 4. CONCLUSION

This study integrated Faster R-CNN with a sparse oblique decision tree as a neuro-symbolic architecture for PCB defect detection on the DeepPCB dataset. The SODT that replaces the classification head preserved the accuracy of Faster R-CNN, with an mAP@0.5 of 0.974 against 0.979 and an mAP@0.5:0.95 of 0.757 against 0.759, while precision rose from 0.910 to 0.923, and the routing margin was essential for this result. With this accuracy preserved, the model produced explanations that are more faithful than Grad-CAM, with a Necessity of 0.768 against 0.126 and a Deletion AUC of 0.200 against 0.599, comparable localization, and a shorter time per image of 80.9 ms against 102.5 ms. The explanation is not a single map but a heatmap for every node on the decision path, each more decisive than random positions at every depth, so each detection can be traced step by step from the context around the defect to its contour. This supports the accountability of detection results and allows technicians to examine the basis of every decision during validation. The study has three limitations: the SODT only mimics the teacher labels, so defect RoIs it fails to mimic add false negatives; the neck was reduced to 64 channels to limit the SODT input; and the explanations were evaluated only quantitatively. Future work should train the SODT directly against the ground truth, use the full 256 neck channels at the cost of a heavier TAO training on 12,544 features, and conduct a user study with PCB inspection technicians to measure the effect of per-node explanations on validation.

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
