ProceedingsofMachineLearningResearchvol284:1–17,202519thConferenceonNeurosymbolicLearningandReasoning
Neurosymbolic models based on
hybrids of convolutional neural networks and decision trees
Rasul Kairgeldin Miguel A´. Carreira-Perpin˜´an
Dept. of Computer Science & Engineering, University of California, Merced
http://eecs.ucmerced.edu
Editors: Leilani H. Gilpin, Eleonora Giunchiglia, PascalHitzler, and Emile van Krieken
Abstract
Buildingonpreviouswork,weproposeaspecificformofneurosymbolicmodelconsistingof
the composition of convolutional neural network layers with a sparse oblique classification
tree (having hyperplane splits using few features). This can be seen as a neural feature
extractionthatfindsamoresuitablerepresentationoftheinputspacefollowedbyaformof
rule-basedreasoningtoarriveatadecisionthatcanbe explained. We showhowtocontrol
thesparsityacrossthedifferentdecisionnodesofthetreeanditseffectontheexplanations
produced. Wedemonstratethisonimageclassificationtasksandshow,amongotherthings,
that relatively small subsets of neurons are entirely responsible for the classification into
specific classes, and that the neurons’ receptive fields focus on areas of the image that
provide best discrimination.
1. Introduction
Neural AI systems and symbolic AI systems have both developed extensively since the
mid-twentieth century and achieved impressive successes in different domains. Symbolic AI
systems, which include many different models (rule-based, logic-based, etc.) are typically
characterized by their transparent nature, in that a human can follow their chain of reason-
ing,atleasttosomeextent. Thisbringscontrolandtrustintheresultsandoftensomeform
ofcorrectnessguarantees (whichiscritical in,say, programverification ortheoremproving).
Neural AI systems, whose most powerful form at present are deep neural nets (NNs), have
excelled at perceptual tasks, such as recognizing patterns in images or generating realistic
text. This is due to their ability to learn from large labeled datasets by optimizing an
objective function that measures the prediction error over the NN parameters. This, in
turn, is made possible by the fact that NNs define differentiable functions, whose gradient
can be used in effective optimization algorithms such as SGD. This also makes it possible
to combine multiple NNs and train them jointly (end-to-end), thanks to the chain rule of
derivatives. On the other hand, neural AI systems typically require large computational
resources in terms of hardware, energy, and training and test time and memory. Also, their
reasoning is impenetrable to humans: even though the structure and parameters can be in-
spected, their size and complexity is such that they behave like a black box. This makes it
hard to understand why they make mistakes and how to correct them, among other things.
Thus, there has long existed an interest in neurosymbolic systems (Hitzler and Sarker,
2021; Kautz, 2022; d’AvilaGarcez andLamb,2023), whichseek tocombinethebestof both
worlds. In particular, there have been previous attempts to combine NNs and trees (see
©2025.

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
section 2). Here, building on the recent work of Hada et al. (2024) (HCZ24 for short), we
will focus on a hybrid model consisting of specific forms of neural and symbolic AI systems:
convolutional NNs (although other NNs could also beused) and decision trees, respectively.
The trees can be turned into a set of decision rules if so desired. The hybrid can be seen as
having the CNN learn some complex features that represent the input (say, an image) in a
way thatismoresuitablefor classification; whiletheclassification is effected by thedecision
tree, whose structure and logic are interpretable by a human (to some extent, depending
on the size of the tree). For example, one can follow the tree reasoning by tracing the path
followed by the input instance from the root to a leaf, solve counterfactual explanations
exactly (Carreira-Perpin˜a´n and Hada, 2021), etc.
Following HCZ24, their procedure starts from a trained CNN and replaces a part or
moduleMofitwithadecisiontreethataimsatbeingapproximately functionally equivalent
(a.f.e.)1. This is achieved by training the tree in a teacher-student way on a dataset with
the inputs that M receives labeled with the corresponding outputs that M produces. For
example, in a LeNet or VGG CNN, M could be all the fully-connected layers that follow
the last convolutional layer (fig. 1). If the predictions of the tree are identical or very close
to those of M on the training and test set, we deem it to be a.f.e. to M. Then, by replacing
Mwiththetree, we defineahybrid,neurosymbolicmodelthatis a.f.e. to theoriginal CNN.
However, thesymbolicmodule(thetreeorsetofrules)bringssomeamountofexplainability
into the hybrid model, which (since the tree represents M well) transfers to the original
CNN.Forexample,amongotherthingsthatHCZ24exploredisthefactthatrelativelysmall
subsets of neuronsare associated with specific classes. This makes it possibleto manipulate
the CNN with surgical precision to force it to make certain classification decisions. The
hybrid model can also replace the original CNN for actual prediction but with much faster
inference.
Key to the success of this approach is the ability of the tree to match the predictive
performance of the module M, while remaining sufficiently interpretable. This may not
always be possible, depending on the complexity of M, but recent advances in decision tree
learning have greatly enlarged the space of tree-type models we can train and improved
the accuracy they can achieve and their scalability to large, high-dimensional datasets. In
particular,wewillusesparse obliquetrees forclassification(Carreira-Perpin˜a´nandTavallali,
2018), whichpredictaconstantlabelateachleaf,butusehyperplanesplitswithfewfeatures
at each decision node. Such trees can be effectively trained with the Tree Alternating
Optimization (TAO) algorithm (Carreira-Perpin˜a´n and Tavallali, 2018). Indeed, as shown
in HCZ24’s and our experiments, the resulting tree is very small (often having just one leaf
per class), yet it produces an a.f.e. hybrid to the original CNN. This would not be possible
with the traditional CART-style decision tree induction algorithms, which use a much more
limited treetype(axis-aligned, usingasinglefeaturepersplit)andamuchmoresuboptimal
training algorithm (greedy recursive partitioning).
In our paper, a first contribution is to make this specific type of model (which would
be a “type-3 or Neuro|Symbolic” system in the classification of Kautz (2022)) known to
1. By this we do not mean the tree represents exactly the same mathematical function as M. This much
stricterrequirementwouldbeoverkillinpracticebecauserealdataoccupyatiny,low-dimensionalregion
of the input space. An estimate of this region is given (indirectly) by the observed data we have for
training and test.
2

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
the neurosymbolic learning community. A second contribution is that we modify the TAO
algorithm so that it learns sparse oblique trees with a controllable sparsity distribution over
the nodes, thus increasing their interpretability while remaining highly accurate. A third
contribution is that we take the interpretability of the hybrid model beyond what HCZ24
did, by showing the receptive fields of neurons that are responsible for the discrimination
between specific classes. We display this as a density map that objectively shows where in
the image those neurons are looking. For example, in the Fashion MNIST dataset we show
how certain neurons act on specific image areas to detect the presence of specific object
parts that are critical to tell one class from another (e.g. to tell a shoe from a bag, a critical
part is the existence of a gap in the shoe front which is occupied by a corner in the bag, and
a certain neuron detects precisely that). Also, using this knowledge, we are able to edit an
image that the CNN misclassifies so it classifies it correctly.
2. Related work
We focus on work involving tree- or rule-based models and neural nets (NNs). Early on,
the black-box nature of NNs was recognized and there were attempts to replace the entire,
trainedNNwithasymbolicrepresentation,specificallyadecisiontreeorasetofrules,which
provided an explainable system. Several approaches of this type were actively researched
in the 1990s and 2000s (reviewed in (Andrews et al., 1995; Duch et al., 2004; Jacobsson,
2005; McCormick et al., 2013; Guidotti et al., 2018)). One approach (Towell and Shavlik,
1993; Fu, 1994; Setiono and Liu, 1996; Baesens et al., 2003; Duch et al., 2004) relied only
on access to the architecture and weight values of a multilayer perceptron (MLP), although
the neuron activations were assumed to be binary. Rules were extracted using some form
of heuristic search. The other is a teacher-student approach, then called “pedagogical”,
which needs a training set (actual or synthetic) on which a decision tree or a set of rules is
trained to mimic the input-output behavior of the MLP (Craven and Shavlik, 1994, 1996;
Domingos, 1998). Although the experiments in these papers were somewhat successful,
they were limited to tiny two-layer MLPs and never scaled up to larger NNs (having more
units and layers). This was due to the use of explainable models of limited power (such as
axis-aligned trees) and/or to the heuristic nature of the procedure (a heuristic search or a
suboptimal training with greedy recursive partitioning algorithms such as CART (Breiman
et al., 1984) or C4.5 (Quinlan, 1993)). In contrast, the work of Hada et al. (2024) did scale
up to larger NNs (LeNet, VGG), without any assumption on the type of activations or NN
architecture other than the ability of the tree to match the accuracy of the NN module
replaced. This was achieved by using a more powerful type of trees (oblique) and a better
optimization (TAO), and also by aiming to replace a part of a NN rather than all of it.
Another related line of work is based on soft decision trees (SDTs) (Jordan and Jacobs,
1994), which use a sigmoid instead of step function at each decision node. Thus, an input
instance traverses all root-leaf paths rather than a single one as in a hard tree, and the
SDT output is a weighted average of all the leaves’ labels. This defines a differentiable
mapping which can be optimized via gradient-based methods, possibly end-to-end together
with other modules (Kontschieder et al., 2015; Good et al., 2023; Hazimeh et al., 2020;
Ibrahim et al., 2024; Borisov et al., 2024). Indeed, a SDT is much closer to (a specific type
of) NNs than to trees. However, the fact that the input instance follows all paths, thus
3

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
y=f(x),
entireneuralnet
y
K output
classes
y=M(z),
x z=F(x), classifierpart
D input F neuralnetfeatures (mimicked
features bytree)
Figure 1: (Adapted from Hada et al. (2021, 2024).) A neurosymbolic model as a CNN-tree
hybrid. The “neural feature” vector z consists of the activations (outputs) of the
F neurons in the last convolutional layer. The fully-connected MLP layers M are
replaced with a sparse oblique classification tree.
touching all parameters, makes the SDT a black box. For the same reason, training and
inference are also much slower than with a hard tree. One can turn a SDT into a hard tree
by replacing each sigmoid with a step function, butthis degrades the accuracy considerably
and is worse than training a hard tree directly (Gazizov et al., 2025).
Finally, appending a tree to a neural net feature extraction has also been done for
other purposes, such as ensembling (Zharmagambetov and Carreira-Perpin˜a´n, 2021) or
compression (Idelbayev et al., 2025).
3. The neural net / tree hybrid model: definition and training
We use the same basic model proposed in HCZ24 and illustrated in fig. 1. The starting
point is a convolutional neural net (CNN), previously trained using a dataset (training and
test) of labeled images (or other type of data) for a classification task. We regard the CNN
as having the form y = M(F(x)), where xis the pixel image, y the predicted label (or label
distribution), F the convolutional layers and M the fully-connected layers (i.e., an MLP).
The partition of the original NN could bedone in other ways, butthis particular one makes
sense in that the output of F (the output of the last convolutional layer, having F neurons)
can be seen as having learned a feature representation2 or embedding, while M acts as a
classifier on them.
Next, the fully-connected layers M are replaced by another classifier, namely a sparse
oblique decision tree y = T(z). This is done in a teacher-student way, by constructing a
new dataset3 having a pair (z ,y′ ) for every original data pair (x ,y ), where z = F(x )
n n n n n n
2. Onecouldalsousenon-adaptivefeaturessuchasSIFT,constructedviaafixedformula,andinterpretable
bydesign.
3. If, instead of approximating the original NN, we seek the best possible hybrid model, we would use
yn ′ =yn instead (i.e., minimize the error on the ground-truth labels). In practice with NNs, especially
4

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
y′
(the output of the last convolutional layer) and = M(z n ) (the output of the CNN). If
n
we can train a tree T such that the error ky′ −T(z )k is very small over the training and
|     |     |     |     | n n |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
test data, then T and T◦F are a.f.e. to M and M◦F, respectively. In HCZ24’s and our
experiments this is the case. The reason is that sparse oblique trees trained with TAO are
| quite powerful | classifiers. |     |     |     |     |     |     |
| -------------- | ------------ | --- | --- | --- | --- | --- | --- |
Finally, we obtain our hybrid CNN-tree model y = T(F(x)), which can be used in
place of the original CNN. Besides providing faster inference, the tree makes it possible to
understand the workings of the CNN features and we explore this in our experiments.
At present, we do not have a way to train the hybrid model end-to-end (i.e., T and
F jointly). This because the tree defines a piecewise constant function, so its gradient is
zero nearly everywhere in parameter space and the chain rule cannot be used to update
F.
However, this is not a limitation if our goal is to understand the meaning and effect on
| classification | of the           | original CNN | features | F.    |            |        |     |
| -------------- | ---------------- | ------------ | -------- | ----- | ---------- | ------ | --- |
| 4. The         | Tree Alternating | Optimization |          | (TAO) | algorithm: | review |     |
Due to space constraints, we keep this brief. More details can be found in the original
| references | (Carreira-Perpin˜a´n | and | Tavallali, | 2018; Hada | et al., 2024). |     |     |
| ---------- | -------------------- | --- | ---------- | ---------- | -------------- | --- | --- |
TreeAlternatingOptimization providesaunifiedframeworkforeffectively trainingcom-
plex decision tree-based models. We will discuss it in the setting of training one oblique
decision tree. Consider a training set {(x ,y )}N ⊂ RD ×{1,...,K} with N samples,
|     |     |     |     | n n n=1 |     |     |     |
| --- | --- | --- | --- | ------- | --- | --- | --- |
D-dimensional features, and K classes. We define an oblique decision tree T(x;Θ) as a
rooted binary tree with decision nodes D and leaf nodes L. Each decision node i ∈ D em-
|     |     |     | (x;θ |     |     | wTx+w |     |
| --- | --- | --- | ---- | --- | --- | ----- | --- |
ploys alinear decision function g i i )to routean instance xto theleft (if i0 ≥ 0)
i
or right child (otherwise), where θ = {w ,w } are learnable parameters. Note how the
|     |     |     | i   | i i0 |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- |
decision function makes hard decisions, unlike in soft trees, where an instance x is propa-
gated to both children with a positive probability. Each leaf j ∈ L contains a constant label
classifier that outputs a single class c ∈ {1,...,K}. We collectively define the parameters
j
of all nodes as Θ = {(w ,w )} ∪ {c } j∈L. The predictive function of the whole tree
|     |     | i i0 | i∈D | j   |     |     |     |
| --- | --- | ---- | --- | --- | --- | --- | --- |
T(x;Θ) then works by routing an instance x to exactly one leaf through a root-leaf path
of (oblique) decision nodes and applying that leaf’s predictor function.
Given a fixed-structure oblique decision tree T(x;Θ) (e.g. a complete tree of depth ∆
or one from CART) with random initial parameters, TAO aims to minimize the following
objective:
N
|     |     | E(Θ) = | L(y | ,T(x ;Θ))+λ | kw k |     | (1) |
| --- | --- | ------ | --- | ----------- | ---- | --- | --- |
|     |     |        | n   | n           | i 1  |     |     |
|     |     |        | n=1 |             | i∈D  |     |     |
|     |     |        | X   |             | X    |     |     |
where L(·,·) is the loss (cross-entropy, 0/1, squared error, etc.) and the term with the
hyperparameter λ ≥ 0 is an ℓ penalty to promote sparsity of the weight vectors.
1
The TAO algorithm relies on two key theorems. The separability condition ensures that
theobjective functionseparates over non-descendantnodes(e.g. allnodesatagiven depth),
allowing for independent and parallel optimization over parameters of each node. The
reduced problem (RP) over a node states that optimizing the objective for a node i∈ D∪L
if overparameterized, this makes little difference because the original NN usually has a very small error
yn ′ =yn
| on thetraining | set, | so for | most training | points. |     |     |     |
| -------------- | ---- | ------ | ------------- | ------- | --- | --- | --- |
5

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
simplifies to a well-defined probleminvolving only thetraining instances reaching that node
(the reduced set (RS), R ⊂ {1,...,N}). We define the following reduced problems. The
i
exact form of the reduced problem differs for leaves and for decision nodes:
• For a decision node i∈ D, the top-level problem of eq. (1) reduces to a weighted 0/1 loss
| binary | classification |     | problem:  |     |        |       |           |     |     |     |
| ------ | -------------- | --- | --------- | --- | ------ | ----- | --------- | --- | --- | --- |
|        |                | E   | (w ,w     |     | L(y ,g | (x ;w | ,w ))+λkw | .   |     |     |
|        |                |     | i i i0 )= |     | n      | i n   | i i0      | i k |     | (2) |
1
∈Ri
n X
Here, y ∈{left ,right }isapseudolabelindicatingtheoptimalchild forx , minimizing
|     | n   | i   | i   |     |     |     |     |     | n   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the subtree loss. The weighted 0/1 loss L(·,·) is defined by the loss difference between
the chosen and alternative child. While optimizing an oblique node is generally NP-hard,
it can be effectively approximated using a surrogate loss like cross-entropy (i.e., logistic
regression). The top-level objective (1) is guaranteed to decrease by accepting updates
| only | if they improve |     | (2), though | this | is often | unnecessary | in practice. |     |     |     |
| ---- | --------------- | --- | ----------- | ---- | -------- | ----------- | ------------ | --- | --- | --- |
• Forleafnodej ∈ L,thetop-levelproblemofeq.(1)reducestoaforminvolvingtheoriginal
lossbutonlyovertheparametersoftheleafpredictorfunction. Itcanbesolvedbyfinding
the majority class (or mean value of the samples in the reduced set for regression)
While these theorems do not prescribe the order in which the nodes should be optimized,
we follow a reverse breadth-first search order: all the nodes at a given depth are optimized
in parallel, starting from the deepest ones until the root. Each optimization subproblem
involves solving either an ℓ 1 -regularized logistic regression or finding a majority class. By
ensuringthatthe(approximate)solutionofthereducedproblemofadecisionnodeimproves
upon the previous node parameter values, TAO is guaranteed to decrease the objective
| function | (1) monotonically. |     |     |     |     |     |     |     |     |     |
| -------- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ℓ
Finally, node pruning occurs automatically because the 1 penalty can drive a node’s
entire weight vector to zero. This makes the node redundant (it sends all instances to the
| same     | child) and | it can | be removed | at the | end.     |     |           |     |     |     |
| -------- | ---------- | ------ | ---------- | ------ | -------- | --- | --------- | --- | --- | --- |
| 5. Finer | sparsity   |        | control    | with a | modified | TAO | algorithm |     |     |     |
The hyperparameter λ controls the overall sparsity in the tree, and by making it large
enough we also achieve pruning (and thus a form of tree structure learning) automatically.
However, it also has the effect that shallow nodes (e.g. the root) are much less sparse than
deeper nodes (e.g. the leaf parents). This is seen in the trained trees and its cause is
explained below. We address this here with a second hyperparameter α that controls how
sparse individual nodes are in relation to the number of instances they handle.
Considerthefollowingobjectivefunction,equalto(1)butwithamodifiedregularization
term (whose motivation is explained later) with hyperparameter α∈ R:
N
|     |     |     |     |     |     |     |     |     | 1, t = 0 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- |
E(Θ) = L(y n ,T(x n ;Θ))+λ h α (|R i |)kw i k , h α (t) = (3)
|     |     |     |     |     |     |     | 1   | (tα, | t > 0 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----- | --- |
|     | n=1 |     |     |     | i∈D |     |     |      |       |     |
|     | X   |     |     |     | X   |     |     |      |       |     |
where R is the RS of node i and |R | its cardinality. This seems difficult to optimize:
|     | i   |     |     | i   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the h term is a non-differentiable function of the tree parameters (specifically, the weight
α
6

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
vectors of the decision nodes in the path ascending from i to the root), because it depends
on |R | (an integer), which depends on the said parameters. However, the TAO theorems
i
still apply, and the only change is in the RP over a decision node i, which now takes the
following form:
E (w ,w ) = L(y ,T(x ;Θ))+λh (|R |)kw k . (4)
i i i0 n n α i i 1
n
X
∈Ri
However, R is constant if the nodes ascending from i are kept fixed (as they are in each
i
TAO iteration). Thus, the RP is exactly as in TAO but with a reweighted hyperparameter
“λh (|R |)”.
α i
The reason for this new algorithm can be seen by dividing4 the RP objective function
by the constant N = |R | (the number of points in i’s RS) and rewriting it as “avg-loss +
i i
λ′ reg”, wherewedefineavg-loss = 1 L(y ,T(x ;Θ)) (theloss perinstance in node
i), reg = kw k , and λ′ = λNα−1 N (a i n e n ff ∈R ec i tive n sparsit n y hyperparameter). This makes it
i 1 i P
clear that α < 1 (e.g. α = 0 as in regular TAO) penalizes larger RSs less than smaller ones
(thus the root weight vector is less sparse); α > 1 penalizes larger RSs more than smaller
ones; and α = 1 penalizes all nodes equally, regardless of how many instances they receive.
This gives us control on how the feature sparsity is distributed across the tree (clearly seen
in fig. 2 vs fig. 7), which is useful for interpretability purposes. Further, it can actually find
trees that are both sparser overall and possibly even more accurate than those of the regular
TAO algorithm. Indeed, experimentally we observe that the trees with best generalization
error occupy a relatively wide region in (λ,α) space, from which we can pick the sparsest
tree.
6. Experiments
In this section, we experimentally demonstrate that our approach learns trees with higher
node sparsity while maintaining, or even improving, accuracy. The resulting tree performs
comparably to neural network. This, along with other experiments, suggests that insights
gainedfromthetreealsoapplytotheneuralnetwork. Importantly, ourfindingsareguaran-
teed to becorrect for hybridmodeland verified to hold well empirically for the original NN;
unlike those of other interpretability attempts based on saliency maps or Shapley values.
Predictive error and tree size We train LeNet-5 on the Fashion MNIST dataset using
PyTorch 1.10. Full hyperparameters and TAO implementation details are provided in the
Appendix. Ourtrees aretrainedon embeddingsfromthelastconvolutional layer (F = 400)
to closely match NN performance. To maintain interpretability, we restrict tree depth to 5.
Thebest-performingtree that uses only 1298 non-zero parameters, achieving E = 5.4%
train
and E = 11.7% is achieved with uniform sparsity distribution (λ = 0.001,α = 1). In
test
comparison, the NN’s fully connected layers contain 59134 parameters—nearly 50 times
more—while improving error rates by only about 1% on both train and test sets.
We also trained axis-aligned trees using CART and TAO. The best CART tree achieved
a training error of 8.7% and a test error of 23.4% with a depth of 31 and 5567 nodes. The
bestTAO univariate treehadatest errorof 21.8% with4400 nodes. Theseresultshighlight
4. This assumes |Ri|>0. If Ri =∅ then h α(|Ri|)=1 and theRP solution is to set wi =0, as in TAO.
7

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
A:60000
kwk0=122
0.69
0 . 4 7 8
0 . 2 6
0.05 − 0 . 2 4
|     |     | B:    | 34 27 4 |     |     | − 0 . 4 2 3 |     | C:    | 25 72 6 |     |
| --- | --- | ----- | ------- | --- | --- | ----------- | --- | ----- | ------- | --- |
|     |     | kw k0 | = 2 19  |     |     | − 0 . 6 1   |     | kw k0 | = 1 43  |     |
0
|     |     |     | 0 . 6 9        |     |     |     |     |     | 0 . 6 9        |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- | --- | -------------- | --- |
|     |     |     | 0 . 4 7 8      |     |     |     |     |     | 0 . 4 7 8      |     |
|     |     |     | 0 . 2 6        |     |     |     |     |     | 0 . 2 6        |     |
|     |     |     | 0.05 − 0 . 2 4 |     |     |     |     |     | 0.05 − 0 . 2 4 |     |
|     |     |     | − 0 . 4 2 3    |     |     |     |     |     | − 0 . 4 2 3    |     |
D: 11 78 7 − 0 . 6 1 E: 22 48 7 F: 12 39 8 − 0 . 6 1 G : 1 3 32 8
| kw k0 = 1 | 93  |     | 0   | kw k0 = | 1 76 |     | kw k0 = 1 48 |     | 0   | k w k = 4 9 |
| --------- | --- | --- | --- | ------- | ---- | --- | ------------ | --- | --- | ----------- |
0
|     | 1 . 0 0 9 |     |     | 9       |     |     | 1 . 0 0 9 |     |     | 1 . 0 0 9 |
| --- | --------- | --- | --- | ------- | --- | --- | --------- | --- | --- | --------- |
|     | 0 . 7 5 8 |     |     | 0 . 4 8 |     |     | 0 . 7 5 8 |     |     | 0 . 7 5 8 |
0 0 . . 2 5 5 0 6 7 0 . 2 6 7 0 0 . . 2 5 5 0 6 7 0 0 . . 2 5 5 0 6 7
|     | 0.005 |     |     | 0.05 |     |     | 0.005 |     |     | 0.005 |
| --- | ----- | --- | --- | ---- | --- | --- | ----- | --- | --- | ----- |
− − 0 0 . . 5 2 0 5 3 4 − 0 . 2 3 4 I : 1 2 063 − − 0 0 . . 5 2 0 5 3 4 − − 0 0 . . 5 2 0 5 3 4
− 0 . 7 5 1 2 H : 1 0 42 4 − 0 . 4 1 2 J: 6 4 39 − 0 . 7 5 1 2 − 0 . 7 5 1 2 K: 7 338
− 1 . 0 0 0 k w k = 8 7 0 k w k 0 = 51 kw k = 32 − 1 . 0 0 0 − 1 . 0 0 0 kw k0 = 66
|     |     |     | 0   |     |     | 0   |         |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- |
|     |     |     | 9   |     | 9   |     | 2 . 0 9 |     |     |     |
61650123456789 56220123456789 0 0 . . 4 6 8 0 0 . . 4 6 8 1 1 . . 0 5 8 59590123456789 59900123456789 0 . 7 5 8 9
|     |     |     | 0 . 2 6 7         |     | 0 . 2 6 7         |     | 0 . 5 6 7         |     |     | 0 0 . . 2 5 5 0 7 |
| --- | --- | --- | ----------------- | --- | ----------------- | --- | ----------------- | --- | --- | ----------------- |
|     |     |     | 0 . 0 4 5         |     | 0 . 0 4 5         |     | 0 . 0 4 5         |     |     | 0 . 0 0 5 6       |
|     |     |     | − − 0 0 . . 4 2 3 |     | − − 0 0 . . 4 2 3 |     | − − 1 0 . . 0 5 3 |     |     | − 0 . 2 5 3 4     |
|     |     |     | − 0 . 6 1 2       |     | − 0 . 6 1 2       |     | − 1 . 5 1 2       |     |     | − 0 . 5 0 2       |
|     |     |     | 0                 |     | 0                 |     | − 2 . 0 0         |     |     | − 0 . 7 5 0 1     |
45300123456789 58940123456789 61470123456789 59160123456789 60800123456789 0123456789 16580123456789 0123456789
|     |     |     |     |     |     |     | 359 |     |     | 5680    |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- |
|     |     |     |     |     |     |     |     |     | λ   | 0.001,α |
Figure 2: Tree trained on LeNet embeddings on Fashion MNIST dataset = = 1.
|     | E   | =     | 5.4%, E | = 11.7%, | number | of non-zero | parameters |     | is 1298. |     |
| --- | --- | ----- | ------- | -------- | ------ | ----------- | ---------- | --- | -------- | --- |
|     |     | train | test    |          |        |             |            |     |          |     |
the limitations of axis-aligned trees in capturing the complex feature interactions learned
by neural networks while also losing interpretability due to their excessive size.
We further analyze the impact of sparsity parameters λ and α. Figure 10 illustrates
|     |     |     |     |     | λ   |     |     | α.  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the regularization path by fixing while gradually increasing As predicted by Eq. 4,
increasing α enhances relative sparsity in decision nodes closer to the root. Beyond a
certain threshold, the root becomes too sparse to sustain an oblique split, causing the tree
to collapse. Lowering λ shifts this threshold, allowing the tree to maintain its structure
over a broader range of α. This effect is evident in the figure: for λ = 100, α =0.5 leads to
collapse, whereas for λ = 1, α = 0.5 still preserves competitive performance. We find that
thereisarangeofλ,αvaluesthatachieve besttreesbutwithdifferentsparsitydistribution,
from which we can pick the best one. For example, comparing 3 trees shows how sparsity
distribution changes from uniform in Fig. 2 to closer to leaves in Fig. 6 and Fig. 7.
Global structure Figure 2 visualizes the structure of the best-performing tree, showing
the learned sparse weights at decision nodes. Each node displays weights in a 4×4 grid,
where each cell represents a 5 × 5 activation from the last convolutional layer of LeNet
(16×5×5). Color coding indicates weight values: red for positive, blue for negative, and
white for zero. At the top of each decision node and the bottom of each leaf, we display the
number of non-zero weights and the size of the reduced set. Additionally, class histograms
are shown on the right side of each node to illustrate the label distribution.
The tree has just 12 leaves, nearly one per class, significantly enhancing interpretability.
Analyzing the leaf nodes reveals a clear hierarchical structure among the classes. The root
node A utilizes only 122 activations to almost perfectly separate classes {9,8,7,5} from
{6,4,2,1,0}, with minor misclassifications in class 3. Similarly, decision node C maintains
comparable sparsity to the root while effectively distinguishingbetween {7,5} and {9,8,3}.
Subtree at F specializes in distinguishing shoe types, and 359 samples (2.89% of the whole
reduced set) of class 8 (“bag”). Interestingly, it uses 148 activations to distinguish class
8

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
1.0
2.0
1.5
0.8
1.0
0.5 0.6
0.0
−0.5 0.4
−1.0
0.2
−1.5
−2.0
0.0
Figure 3: Whereisadecisionnodelookingatintheimage? Plot1showsthesparseweights
ofdecisionnodeJforeach CNNfeaturemapinthelastlayer ofLeNet(16×5×5).
Plots 2–3 show the receptive field produced by neurons with non-zero decision
node weights on the mean image from left and right leaf. Receptive fields follow
the order of CNN outputs left to right and top to bottom. Color and thickness
of receptive fields correspond to weights of decision node. Plot 4 is a heatmap of
the “density” of the receptive fields. Tree hyperparameters: λ = 0.001, α = 1.
5 (“sandal”) from both “bags” and “sneakers”, while differentiating between “sneakers”
and “bags” uses relatively small subset of neurons (only 32 out of 400). Similarly, subtree
G focuses on distinguishing classes {9,8,3}. Unlike F, subtree G) uses significantly fewer
activations in decision nodes G (49) and K (66). These examples highlight how certain
neurons specialize in recognizing specific patterns, allowing the sparse tree to retain only
essential activations for classification. In contrast, decision node D requires 193 activations
to differentiate class 4 (“coat”) from class 6 (“shirt”). Subtree E is similar to subtree F as
it uses more information to separate sets of classes ({3,2} and {0,1}), but distinguishing
within these sets requires far fewer information. Visual inspection suggests that the high
similarity between theseclasses forcesthetreetousemoreactivations tomaintain accuracy.
Where is a specific neuron, and a specific decision node, looking at? Although
understanding the precise meaning of a neural feature is difficult, due to the complexity
of the function of the pixels it defines, what is possible is to construct its receptive field
(RF). This is the region of the image that a neuron in a convolutional layer is looking at. It
occurs by design: a neuron in convolutional layer i only receives input from a small subset
of neurons in layer i−1. Also, since those neurons are organized spatially in a systematic
way over the image grid, so are their RFs. We can construct the RFs for all F neurons and
then look at how they participate in a decision node. Fig. 3 shows the RFs for a selected
node, with a thickness proportional to their weight in the node, overlaid on an inputimage.
Interestingly, RFs associated with the highest positive weights are concentrated in the top
left region of the image. This suggests that neurons identifying ink presence in this area
play a crucial role in distinguishing between a bag and a shoe, as the shoe image typically
lacks content in that specific RF location.
Also, for any decision node, we can construct a RF density map over the image as a
linear combination of all the F RFs (considered as 0/1 indicator functions over the image)
using the magnitude of their weights in the node. This objectively indicates the region of
the image that is being used in that node: zero density means it is not used at all, and
the larger the density the more it is used (because many neurons look there and/or their
9

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
Figure 4: Sample of class 8 misclassified as 7 (Left). Receptive field of neurons from the
last convolutional layer of LeNet with largest positive (red) and negative (blue)
weights in oblique decision node J (Middle). Small changes in the intersection of
two regions fixed the misclassification error (Right).
weights are large). In the example in fig. 3, this clearly shows that that node focuses on the
region where typically we find either the hollow in the front of a shoe or the top-left corner
of a bag, which (at that point in the tree) is sufficient to discriminate those two classes.
Weemphasizethat1)thiswouldnotbepossiblewithoutthetreeandtheweightvectors,
and 2) it is much more objective than saliency maps and other NN visualization techniques
that have been shown to beoften misleading (Fong andVedaldi, 2017; Adebayo et al., 2018;
Ghorbani et al., 2019).
Correcting classification mistakes of the neural net The above information can be
used to understand why an image is (mis)classified as a certain class, and even to alter this
by editing the image. Fig. 4 shows this with the image of a bag that the NN (and the tree)
misclassifies as a shoe. From the RF density map discussed above it seems like the bag,
which has an odd shape, is missing the typical left corner that bags have. When we edit
the image to add that, both the NN and the tree classify it correctly.
7. Conclusion
Wehaverevisited,andintroducedtotheneurosymboliclearningcommunity,ahybridmodel
consisting of a convolutional feature extraction module composed with a sparse oblique
decision tree. By introducing a new type of regularization and suitably modifying the
Tree Alternating Optimization (TAO) algorithm, we have further developed this model to
control how the feature sparsity is distributed over the tree structure. We train it in two
stages: first, we train a regular CNN using SGD; then, we train the tree to replace its fully-
connected layers using a teacher-student approach and our modified TAO algorithm. This
produces an accurate hybrid model that benefits from the ability of convolutional layers
to learn a better representation of an image, and from the ability of trees to explain the
reasoning used to classify the image based on those neural features. This makes it possible
to understand to some extent which neurons affect which classes, where in the image those
neurons are looking at, why a specific image is (mis)classified as a certain class, and how
to edit an image to alter its classification in desired ways.
10

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
Acknowledgments
| Work partially | supported | by NSF | award IIS–2007147. |     |
| -------------- | --------- | ------ | ------------------ | --- |
References
Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been
Kim. Sanity checks for saliency maps. In S. Bengio, H. Wallach, H. Larochelle, K. Grau-
man, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Pro-
cessing Systems (NeurIPS), volume 31, pages 9505–9515. MIT Press, Cambridge, MA,
2018.
Robert Andrews, Joachim Diederich, and Alan B. Tickle. Survey and critique of techniques
for extracting rules from trained artificial neural networks. Knowledge-Based Systems, 8
| (6):373–389, | December | 1995. |     |     |
| ------------ | -------- | ----- | --- | --- |
Bart Baesens, Rudy Setiono, Christophe Mues, and Jan Vanthienen. Using neural network
rule extraction and decision tables for credit-risk evaluation. Management Science, 49
| (3):255–350, | March | 2003. |     |     |
| ------------ | ----- | ----- | --- | --- |
Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and
Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE Trans. Neural
| Networks | and Learning | Systems, | 35(6):7499–7519, | June 2024. |
| -------- | ------------ | -------- | ---------------- | ---------- |
Leo J. Breiman, Jerome H. Friedman, R. A. Olshen, and Charles J. Stone. Classification
| and Regression | Trees. | Wadsworth, | Belmont, Calif., | 1984. |
| -------------- | ------ | ---------- | ---------------- | ----- |
A´.
Miguel Carreira-Perpin˜a´n and Suryabhan Singh Hada. Counterfactual explanations for
oblique decision trees: Exact, efficient algorithms. In Proc. of the 35th AAAI Conference
on Artificial Intelligence (AAAI 2021), pages 6903–6911, Online, February 2–9 2021.
MiguelA´.Carreira-Perpin˜a´nandPooyaTavallali.
Alternatingoptimizationofdecisiontrees,
with application to learningsparseoblique trees. In S.Bengio, H.Wallach, H. Larochelle,
K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information
Processing Systems (NeurIPS),volume31,pages1211–1221. MITPress,Cambridge,MA,
2018.
Mark Craven and Jude W. Shavlik. Using sampling and queries to extract rules from
trained neural networks. In Proc. of the 11th Int. Conf. Machine Learning (ICML’94),
| pages 37–45, | 1994. |     |     |     |
| ------------ | ----- | --- | --- | --- |
Mark Craven and Jude W. Shavlik. Extracting tree-structured representations of trained
networks. In David S. Touretzky, M. C. Mozer, and M. E. Hasselmo, editors, Advances
in Neural Information Processing Systems (NIPS), volume 8, pages 24–30. MIT Press,
| Cambridge, | MA, 1996. |     |     |     |
| ---------- | --------- | --- | --- | --- |
Artur d’Avila Garcez and Lu´ıs C. Lamb. Neurosymbolic AI: The 3rd wave. Artificial
| Intelligence | Review, | 56:12387–12406, | 2023. |     |
| ------------ | ------- | --------------- | ----- | --- |
11

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
Pedro Domingos. Knowledge discovery via multiple models. Intelligent Data Analysis, 2
| (1–4):187–202, |     | 1998. |     |     |     |     |
| -------------- | --- | ----- | --- | --- | --- | --- |
W lodzis lawDuch,RudySetiono,andJacekM.Z˙urada.Computationalintelligencemethods
for rule-based data understanding. Proc. IEEE, 92(5):771–805, May 2004.
Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, and Chih-Jen Lin. LI-
BLINEAR: A library for large linear classification. J. Machine Learning Research, 9:
| 1871–1874, | August | 2008. |     |     |     |     |
| ---------- | ------ | ----- | --- | --- | --- | --- |
RuthC.Fong andAndreaVedaldi. Interpretableexplanations of black boxes by meaningful
perturbation. In Proc. 16th Int. Conf. Computer Vision (ICCV’17), pages 3449–3457,
| Venice, | Italy, | December | 11–18 | 2017. |     |     |
| ------- | ------ | -------- | ----- | ----- | --- | --- |
LiMin Fu. Rule generation from neural networks. IEEE Trans. Systems, Man, and Cyber-
| netics, | 24(8):1114–1124, |     | August | 1994. |     |     |
| ------- | ---------------- | --- | ------ | ----- | --- | --- |
Kuat Gazizov, Arman Zharmagambetov, and Miguel A´. Carreira-Perpin˜a´n. A critical com-
| parison | of soft | vs hard | oblique | classification | trees. arXiv, | 2025. |
| ------- | ------- | ------- | ------- | -------------- | ------------- | ----- |
Amirata Ghorbani, Abubakar Abid, and James Zou. Interpretation of neural network is
fragile. In Proc. of the 33rd AAAI Conference on Artificial Intelligence (AAAI 2019),
| pages 3681–3688, |     | Honolulu, |     | HI, January | 27 – February | 1 2019. |
| ---------------- | --- | --------- | --- | ----------- | ------------- | ------- |
Jack Good, Torin Kovach, Kyle Miller, and Artur Dubrawski. Feature learning for inter-
pretable, performant decision trees. In A. Oh, T. Naumann, A. Globerson, K. Saenko,
M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems
(NeurIPS), volume 36, pages 66571–66582. MIT Press, Cambridge, MA, 2023.
RiccardoGuidotti, AnnaMonreale, SalvatoreRuggieri, FrancoTurini,FoscaGiannotti, and
Dino Pedreschi. A survey of methods for explaining black box models. ACM Computing
| Surveys, | 51(5):93, | May | 2018. |     |     |     |
| -------- | --------- | --- | ----- | --- | --- | --- |
SuryabhanSinghHada,MiguelA´.Carreira-Perpin˜a´n,andArmanZharmagambetov.
Sparse
oblique decision trees: A tool to understand and manipulate neural net features.
| arXiv:2104.02922, |     | April | 7 2021. |     |     |     |
| ----------------- | --- | ----- | ------- | --- | --- | --- |
SuryabhanSinghHada,MiguelA´.Carreira-Perpin˜a´n,andArmanZharmagambetov.
Sparse
oblique decision trees: A tool to understand and manipulate neural net features. Data
| Mining | and Knowledge |     | Discovery, | 38:2863–2902, | 2024. |     |
| ------ | ------------- | --- | ---------- | ------------- | ----- | --- |
Hussein Hazimeh, Natalia Ponomareva, Petros Mol, Zhenyu Tan, and Rahul Mazumder.
The tree ensemble layer: Differentiability meets conditional computation. In Hal
Daum´eIIIandAartiSingh,editors,Proc. of the 37th Int. Conf. Machine Learning (ICML
| 2020), | pages 4138–4148, |     | Online, | July 13–18 | 2020. |     |
| ------ | ---------------- | --- | ------- | ---------- | ----- | --- |
PascalHitzlerandMdKamruzzamanSarker,editors. Neuro-Symbolic ArtificialIntelligence:
The State of the Art. Number 342 in Frontiers in Artificial Intelligence and Applications.
| IOS Press, | 2021. |     |     |     |     |     |
| ---------- | ----- | --- | --- | --- | --- | --- |
12

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
Shibal Ibrahim, Kayhan Behdin, and Rahul Mazumder. End-to-end feature selection ap-
proach for learning skinny trees. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen
Li, editors, Proc. of the 27th Int. Conf. Artificial Intelligence and Statistics (AISTATS
| 2024), pages | 2863–2871, |     | Valencia, |     | Spain, May | 2–4 2024. |
| ------------ | ---------- | --- | --------- | --- | ---------- | --------- |
Yerlan Idelbayev, Arman Zharmagambetov, Magzhan Gabidolla, and Miguel A´. Carreira-
Perpin˜a´n. Faster neural net inference via forests of sparse oblique decision trees. arXiv,
2025.
HenrikJacobsson. Ruleextraction fromrecurrentneuralnetworks: Ataxonomyandreview.
| Neural | Computation, |     | 17(6):1223–1263, |     | June | 2005. |
| ------ | ------------ | --- | ---------------- | --- | ---- | ----- |
Michael I. Jordan and Robert A. Jacobs. Hierarchical mixtures of experts and the EM
| algorithm. | Neural | Computation, |     |     | 6(2):181–214, | March 1994. |
| ---------- | ------ | ------------ | --- | --- | ------------- | ----------- |
Henry Kautz. The third AI summer. AI Magazine, 43(1):105–125, Spring 2022.
Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, and Samuel Rota Bul´o. Deep
neural decision forests. In Proc. 15th Int. Conf. Computer Vision (ICCV’15), pages
| 1467–1475, | Santiago, |     | Chile, | December | 11–18 | 2015. |
| ---------- | --------- | --- | ------ | -------- | ----- | ----- |
Keith McCormick, Dean Abbott, Meta S. Brown, Tom Khabaza, and Scott R. Mutchler.
| IBM SPSS | Modeler | Cookbook. |     | Packt | Publishing, | 2013. |
| -------- | ------- | --------- | --- | ----- | ----------- | ----- |
Fabian Pedregosa, Ga¨el Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand
Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent
Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher,
Matthieu Perrot, and E´douard Duchesnay. Scikit-learn: Machine learning in Python.
J. Machine Learning Research, 12:2825–2830, October 2011. Available online at
https://scikit-learn.org.
J. Ross Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.
Rudy Setiono and Huan Liu. Symbolic representation of neural networks. IEEE Computer,
| 29(3):71–77, | March | 1996. |     |     |     |     |
| ------------ | ----- | ----- | --- | --- | --- | --- |
Geoffrey G. Towell and Jude W. Shavlik. Extracting refined rules from knowledge-based
| neural networks. |     | Machine | Learning, |     | 13(1):71–101, | October 1993. |
| ---------------- | --- | ------- | --------- | --- | ------------- | ------------- |
Arman Zharmagambetov and Miguel A´. Carreira-Perpin˜a´n. A simple, effective way to
improve neural net classification: Ensembling unit activations with a sparse oblique de-
cision tree. In IEEE Int. Conf. Image Processing (ICIP 2021), pages 369–373, Online,
| September | 19–22 | 2021. |     |     |     |     |
| --------- | ----- | ----- | --- | --- | --- | --- |
13

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
Appendix A. Appendix
A.1. Setup
We implemented TAO in Python 3.11. To solve reduced problem in the decision nodes
we used scikit-learn logistic regression (Pedregosa et al., 2011) with LIBLINEAR (Fan
etal.,2008). Theregularization pathwasconstructed byinitializing treewithhighdifferent
value of λ and increasing α from negative values (typically −1) to 1 with a small step. We
implemented LeNet in Pytorch 1.10. We trained it with Adam optimizer with learning
rate of 0.001, and weight decay 1e−4 on NVIDIA TITAN X. All tree experiments were
conducted on the machine Intel Xeon CPU E5-2699 v3 @ 2.30GHz, 256 GB RAM.
A.2. Masking
Fig. 9 was produced similar to HCZ24 by extracting a mask of non-zero entries from the
sparse weight vector, then multiplying it element-wise with the activations from the last
CNN layers. The result is passed through fully connected layer to produce classification.
More details are described by Hada et al. (2024). The results show that the tree is able
to capture specialized neurons, by keeping which, NN is only able to classify some specific
classes.
|       |          |     |     | )}N    |     | RD         |     |     |
| ----- | -------- | --- | --- | ------ | --- | ---------- | --- | --- |
| input | training | set | {(x | n ,y n | ⊂   | ×{1,...,K} |     |     |
n=1
|     | initial         | tree | T   |        |     |     |     |     |
| --- | --------------- | ---- | --- | ------ | --- | --- | --- | --- |
|     | hyperparameters |      |     | λ ≥ 0, | α ∈ | R   |     |     |
repeat
|     | for i     |                | T,      |       |            |         |       |     |
| --- | --------- | -------------- | ------- | ----- | ---------- | ------- | ----- | --- |
|     | ∈ nodes   | of             | visited |       | in reverse | BFS     |       |     |
|     | if i is a | leaf then      |         |       |            |         |       |     |
|     | θ ←       | majority-class |         | label | in the     | reduced | set R |     |
|     | i         |                |         |       |            |         | i     |     |
else
|     | generate | pseudolabels |           | y   | for each | instance        | n ∈R |      |
| --- | -------- | ------------ | --------- | --- | -------- | --------------- | ---- | ---- |
|     |          |              |           |     | n        |                 |      | i    |
|     | (w ,w    | ) ←          | minimizer |     | of the   | reduced problem | (eq. | (4)) |
i i0
| until | stop |     |     |     |     |     |     |     |
| ----- | ---- | --- | --- | --- | --- | --- | --- | --- |
T:
| postprocess |     | remove |     | dead branches |     | & pure subtrees |     |     |
| ----------- | --- | ------ | --- | ------------- | --- | --------------- | --- | --- |
| return      | T   |        |     |               |     |                 |     |     |
Figure 5: (Adapted from Hada et al. (2021, 2024).) Pseudocode for the tree alternating
optimization(TAO)algorithm,modifiedtohandleournewsparsityregularization
R.
term with hyperparameter α ∈ Visiting each node in reverse breadth-first
search (BFS) order means scanning depths from depth(T) down to 1, and at
each depth processing (in parallel, if so desired) all nodes at that depth. “stop”
occurs when either the objective function decreases less than a set tolerance or
| the number | of iterations |     | reaches | a   | set limit. |     |     |     |
| ---------- | ------------- | --- | ------- | --- | ---------- | --- | --- | --- |
14

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
A:60000
kwk0=253
0.69
0.47 8
0.26
5
0.04
- 0 . 2 2 3
|     |          |     |           | B:28878  |             |     |             |     |     | - 0 . 4 1 |         |           |     | C:31112  |             |     |     |           |     |
| --- | -------- | --- | --------- | -------- | ----------- | --- | ----------- | --- | --- | --------- | ------- | --------- | --- | -------- | ----------- | --- | --- | --------- | --- |
|     |          |     |           | kwk0=262 |             |     |             |     |     | -0.60     |         |           |     | kwk0=248 |             |     |     |           |     |
|     |          |     |           |          | 9           |     |             |     |     |           |         |           |     |          | 0 . 6 9     |     |     |           |     |
|     |          |     |           |          | 0.47 8      |     |             |     |     |           |         |           |     |          | 0 . 4 8     |     |     |           |     |
|     |          |     |           |          | 0.26        |     |             |     |     |           |         |           |     |          | 0.26 7      |     |     |           |     |
|     |          |     |           |          | 5           |     |             |     |     |           |         |           |     |          | 5           |     |     |           |     |
|     |          |     |           |          | 0.04        |     |             |     |     |           |         |           |     |          | 0.04        |     |     |           |     |
|     |          |     |           |          | - 0 . 2 3   |     |             |     |     |           |         |           |     |          | - 0 . 2 3   |     |     |           |     |
|     |          |     |           |          | - 0 . 4 1 2 |     |             |     |     |           |         |           |     |          | - 0 . 4 1 2 |     |     |           |     |
|     | D:18284  |     |           |          | 0           |     | E:10594     |     |     |           | F:17011 |           |     |          | -0.60       |     |     | G:14111   |     |
|     | kwk0=105 |     |           |          |             |     | kwk0=190    |     |     |           | kwk0=55 |           |     |          |             |     |     | kwk0=124  |     |
|     |          |     | 0 . 8 9   |          |             |     | 1 . 0 0 9   |     |     |           |         | 9         |     |          |             |     |     | 1.08 9    |     |
|     |          |     | 0 . 6 8   |          |             |     | 0 . 7 5 8   |     |     |           |         | 1.08      |     |          |             |     |     |           |     |
|     |          |     | 0 . 4 6 7 |          |             |     | 0 . 5 0 6 7 |     |     |           |         | 0 . 5 6 7 |     |          |             |     |     | 0 . 5 6 7 |     |
|     |          |     | 0 . 2 5   |          |             |     | 0 . 2 5 5   |     |     |           |         | 5         |     |          |             |     |     | 5         |     |
|     |          |     | 0 . 0 4   |          |             |     | 0 . 0 0 4   |     |     |           |         | 0 . 0 4   |     |          |             |     |     | 0 . 0 4   |     |
|     |          |     | - 0 . 2 3 |          |             |     | - 0 . 2 5 3 |     |     |           |         | - 0 . 5 3 |     |          |             |     |     | - 0 . 5 3 |     |
- 0 . 4 1 2 - - 0 0 . . 7 5 5 0 1 2 1 2 J : 5 09 1 K: 8 170 1 2
H: 12 37 0 - - 0 0 . . 8 6 0 - 1 . 0 0 0 I : 1 1 920 - 1 . 0 0 kw k = 1 21 kw - 1 . 0 0
| 15 kw | k0 = 1 39 |               |      |      |     |      |     |      | k w  | k 0 = 89 |               |      |           | 0    |     | k0 = 20 |           |      |      |
| ----- | --------- | ------------- | ---- | ---- | --- | ---- | --- | ---- | ---- | -------- | ------------- | ---- | --------- | ---- | --- | ------- | --------- | ---- | ---- |
|       |           | 9             |      |      |     |      |     |      |      |          | 9             |      | 9         |      |     |         | 9         |      |      |
|       |           | 0 0 . . 6 8 8 |      | 9    |     | 9    |     | 9    |      |          | 0 0 . . 6 8 8 |      | 1 . 0 8   |      |     |         | 1 . 5 8   |      | 9    |
|       |           | 0 . 4 7       |      | 8    |     | 8    |     | 8    |      |          | 0 . 4 7       |      | 0 . 5 7   |      |     |         | 1 . 0 7   |      | 8    |
|       |           | 0 . 2 6       |      | 6 7  |     | 6 7  |     | 6 7  |      |          | 0 . 2 6       |      | 6         |      |     |         | 0 . 5 6   |      | 6 7  |
|       |           | 0 . 0 4 5     |      | 4 5  |     | 4 5  |     | 4 5  |      |          | 0 . 0 4 5     |      | 0 . 0 4 5 |      |     |         | 0 . 0 4 5 |      | 4 5  |
|       |           | - 0 . 2 3     |      | 3    |     | 3    |     | 3    |      |          | - 0 . 2 3     |      | 3         |      |     |         | - 0 . 5 3 |      | 3    |
|       |           | - 0 . 4 2     |      | 1 2  |     | 1 2  |     | 1 2  |      |          | - 0 . 4 2     |      | - 0 . 5 2 |      |     |         | - 1 . 0 2 |      | 1 2  |
|       |           | - 0 . 6 1     |      | 0    |     | 0    |     | 0    |      |          | - 0 . 6 1     |      | - 1 . 0 1 |      |     |         | - 1 . 5 1 |      | 0    |
|       |           | - 0 . 8 0     |      | 5914 |     | 5454 |     | 5140 |      |          | - 0 . 8 0     |      | 0         |      |     |         | 0         |      | 5941 |
|       |           | 9 8           |      | 9 8  |     |      |     |      |      | 9 8      | 9 8           |      | 9 8       |      | 9 8 |         | 9 8       | 9 8  |      |
|       |           | 7 6           |      | 7 6  |     |      |     |      |      | 7 6      | 7 6           |      | 7 6       |      | 7 6 |         | 7 6       | 7 6  |      |
|       |           | 5             |      | 5    |     |      |     |      |      | 5        | 5             |      | 5         |      | 5   |         | 5         | 5    |      |
|       |           | 4 3           |      | 4 3  |     |      |     |      |      | 4 3      | 4 3           |      | 4 3       |      | 4 3 |         | 4 3       | 4 3  |      |
|       |           | 2 1           |      | 2 1  |     |      |     |      |      | 2 1      | 2 1           |      | 2 1       |      | 2 1 |         | 2 1       | 2 1  |      |
|       |           | 0             |      | 0    |     |      |     |      |      | 0        | 0             |      | 0         |      | 0   |         | 0         | 0    |      |
|       | 6240      |               | 6130 |      |     |      |     |      | 5973 |          | 5947          | 1030 |           | 4061 |     | 6107    |           | 2063 |      |
Figure 6: Tree trained on LeNet embeddings on Fashion MNIST dataset λ = 1,α = 0.25. E = 4.4%, E = 12.4%, number
|     |     |     |          |            |     |          |     |     |     |     |     |     |     | train |     | test |     |     |     |
| --- | --- | --- | -------- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | ---- | --- | --- | --- |
|     |     | of  | non-zero | parameters |     | is 1606. |     |     |     |     |     |     |     |       |     |      |     |     |     |

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
A:60000
kwk0=341
0.69
0.48 7
0.26
0.05
− 0 . 2 4
|     |     |     | B:25404  |     |     |     |     |     | 3         |     |     |     | C:34596  |     |     |     |     |
| --- | --- | --- | -------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | -------- | --- | --- | --- | --- |
|     |     |     | kwk0=309 |     |     |     |     |     | − 0 . 4 2 |     |     |     | kwk0=235 |     |     |     |     |
−0.6 0 1
|     |     |     |     | 0 . 6     |     |     |     |     |     |     |     |     |     | 0 . 8     |     |     |     |
| --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- |
|     |     |     |     | 8 9       |     |     |     |     |     |     |     |     |     | 0 . 6 8 9 |     |     |     |
|     |     |     |     | 0 . 4 7   |     |     |     |     |     |     |     |     |     | 0 . 4 7   |     |     |     |
|     |     |     |     | 0 . 2 6   |     |     |     |     |     |     |     |     |     | 0 . 2 6   |     |     |     |
|     |     |     |     | 0.05      |     |     |     |     |     |     |     |     |     | 0.05      |     |     |     |
|     |     |     |     | − 0 . 2 4 |     |     |     |     |     |     |     |     |     | − 0 . 2 4 |     |     |     |
|     |     |     |     | − 0 . 4 3 |     |     |     |     |     |     |     |     |     | − 0 . 4 3 |     |     |     |
D: 11 54 0 − 0 . 6 1 2 E : 1 3 86 4 F: 21 24 9 − 0 . 6 1 2 G: 13 34 7
|     |           |               |     | 0   |     | k w k       | = 9 3 |     |     |           |             |     |     | − 0 . 8 0 |     |              |     |
| --- | --------- | ------------- | --- | --- | --- | ----------- | ----- | --- | --- | --------- | ----------- | --- | --- | --------- | --- | ------------ | --- |
| kw  | k0 = 2 19 |               |     |     |     | 0           |       |     |     | kw k0 = 2 | 35          |     |     |           |     | kw k0 = 1 04 |     |
|     |           | 0 . 7 5 9     |     |     |     | 9           |       |     |     |           | 0 . 8 9     |     |     |           |     | 0 . 8 9      |     |
|     |           | 8             |     |     |     | 1 . 0 8     |       |     |     |           | 0 . 6 8     |     |     |           |     | 0 . 6 8      |     |
|     |           | 0 . 5 0 7     |     |     |     | 0 . 5 7     |       |     |     |           | 0 . 4 7     |     |     |           |     | 0 . 4 7      |     |
|     |           | 0 . 2 5 6     |     |     |     | 6           |       |     |     |           | 0 . 2 6     |     |     |           |     | 0 . 2 6      |     |
|     |           | 0.005         |     |     |     | 0.05        |       |     |     |           | 0.05        |     |     |           |     | 0.05         |     |
|     |           | − 0 . 2 5 3 4 |     |     |     | − 0 . 5 3 4 |       |     |     |           | − 0 . 2 3 4 |     |     |           |     | − 0 . 2 3 4  |     |
− 0 . 5 0 2 2 I : 1 0 792 − 0 . 4 2 J : 1 0 45 7 K: 7 436 − 0 . 4 2
− 0 . 7 5 1 − 1 . 0 1 H : 1 1 97 3 − 0 . 6 1 k w k = 4 8 kw − 0 . 6 1
16 0 0 k w k = 3 2 k w k 0 = 60 − 0 . 8 0 0 k0 = 25 − 0 . 8 0
0
|     |     |     |     |     |     |     |             |     |     | 1 . 5     |     |     | 1 . 0 0     |     |             |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --------- | --- | --- | ----------- | --- | ----------- | --- | --- |
|     |     |     |     |     | 9   |     | 0 . 7 5 9   |     |     | 1 . 0 8 9 |     |     | 0 . 7 5 8 9 |     | 0 . 7 5 8 9 |     | 9   |
|     | 8 9 |     | 8 9 |     | 8   |     | 0 . 5 0 8   |     |     | 7         |     |     | 0 . 5 0 7   |     | 0 . 5 0 7   |     | 8   |
|     | 7   |     | 7   |     | 6 7 |     | 0 . 2 5 6 7 |     |     | 0 . 5 6   |     |     | 0 . 2 5 6   |     | 0 . 2 5 6   |     | 6 7 |
|     | 5 6 |     | 5 6 |     | 5   |     | 0 . 0 0 5   |     |     | 0 . 0 5   |     |     | 0 . 0 0 5   |     | 0 . 0 0 5   |     | 5   |
|     | 4   |     | 4   |     | 3 4 |     | − 0 . 2 5 4 |     |     | − 0 . 5 4 |     |     | − 0 . 2 5 4 |     | − 0 . 2 5 4 |     | 3 4 |
|     | 2 3 |     | 2 3 |     | 2   |     | − 0 . 5 0 3 |     |     | 3         |     |     | − 0 . 5 0 3 |     | − 0 . 5 0 3 |     | 2   |
1 1 18910 1 − 0 . 7 5 2 − 1 . 0 1 2 − 0 . 7 5 1 2 − 0 . 7 5 1 2 59110 1
|     | 56010 |     | 59390 |     |     |     | 1   |     |     | − 1 . 5 0 |     |     | − 1 . 0 0 0 |     | 0   |     |     |
| --- | ----- | --- | ----- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | ----------- | --- | --- | --- | --- |
0
|     |     |     |     |     |         |     |        | 9     |     | 9     |     | 9     |     | 9     | 9     |     | 9    |
| --- | --- | --- | --- | --- | ------- | --- | ------ | ----- | --- | ----- | --- | ----- | --- | ----- | ----- | --- | ---- |
|     |     |     |     |     | 8 9     |     | 8 9    | 8     |     | 8     |     | 8     |     | 8     | 8     |     | 8    |
|     |     |     |     |     | 7       |     | 7      | 6 7   |     | 6 7   |     | 6 7   |     | 6 7   | 6 7   |     | 6 7  |
|     |     |     |     |     | 5 6     |     | 5 6    | 5     |     | 5     |     | 5     |     | 5     | 5     |     | 5    |
|     |     |     |     |     | 4       |     | 4      | 3 4   |     | 3 4   |     | 3 4   |     | 3 4   | 3 4   |     | 3 4  |
|     |     |     |     |     | 2 3     |     | 2 3    | 1 2   |     | 1 2   |     | 1 2   |     | 1 2   | 1 2   |     | 1 2  |
|     |     |     |     |     | 59750 1 |     | 5990 1 | 48120 |     | 59800 |     | 63770 |     | 40800 | 61630 |     | 1270 |
|     |     |     |     |     |         |     | 8      |       |     |       |     |       |     |       |       |     | 3    |
Figure 7: Tree trained on LeNet embeddings on Fashion MNIST dataset, λ = 100,α = −0.25. E = 5.2%, E = 13.0%,
|     |     |        |             |            |     |     |       |     |     |     |     |     |     | train |     | test |     |
| --- | --- | ------ | ----------- | ---------- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | ----- | --- | ---- | --- |
|     |     | number | of non-zero | parameters |     | is  | 1701. |     |     |     |     |     |     |       |     |      |     |

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
1.0
0.75
|     | 0.50 |     |     |     |     | 0.8 |
| --- | ---- | --- | --- | --- | --- | --- |
0.25
0.6
0.00
−0.25
|     | −0.50 |     |     |     |     | 0.4 |
| --- | ----- | --- | --- | --- | --- | --- |
−0.75
0.2
Figure 8: Whereisadecisionnodelookingatintheimage? Plot1showsthesparseweights
ofdecisionnodeHforeachCNNfeaturemapinthelastlayerofLeNet(16×5×5).
Plots 2–3 show the receptive field produced by neurons with non-zero decision
node weights on the mean image from left and right leaf. Receptive fields follow
the order of CNN outputs left to right and top to bottom. Color and thickness
of receptive fields correspond to weights of decision node. Plot 4 is a heatmap of
the “density” of the receptive fields. Tree hyperparameters: λ =100, α= −0.25.
0 0 1 2 3 4 5 6 7 8 9 0 0 1 2 3 4 5 6 7 8 9 0 0 1 2 3 4 5 6 7 8 9 0 0 1 2 3 4 5 6 7 8 9 0 0 1 2 3 4 5 6 7 8 9 0 0 1 2 3 4 5 6 7 8 9
| 1   | 1   | 1   | 1   | 1   | 1   | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 2   | 2   | 2   | 2   | 2   | 2   | 0.8 |
| 3   | 3   | 3   | 3   | 3   | 3   |     |
| 4   | 4   | 4   | 4   | 4   | 4   | 0.6 |
| 5   | 5   | 5   | 5   | 5   | 5   |     |
| 6   | 6   | 6   | 6   | 6   | 6   | 0.4 |
| 7   | 7   | 7   | 7   | 7   | 7   |     |
| 8   | 8   | 8   | 8   | 8   | 8   | 0.2 |
| 9   | 9   | 9   | 9   | 9   | 9   |     |
0.0
Figure 9: Confusionmatrixofneuralnetworkpredictionsmodifiedbytreemask. Themasks
are produced by analyzing sparse weights of decision nodes. From left to right
|     |     |     |     | λ   | α −0.25. |     |
| --- | --- | --- | --- | --- | -------- | --- |
first 4 are decision nodes G, H, I, J of tree with = 100, = Next,
decision node K of tree with λ = 1, α = 0.25. Lastly, positive weights from
| subtree | G in tree | with λ = 0.001, | α= 1. The | values | are normalized. |     |
| ------- | --------- | --------------- | --------- | ------ | --------------- | --- |
17

Neurosymbolicmodelsbasedonhybridsofconvolutionalneuralnetworksanddecisiontrees
5
| Etrain |     | 30  |     | )%(.smaraporez-non# 80 |     |
| ------ | --- | --- | --- | ---------------------- | --- |
80
| Etest |     | 25  |     | 4   |     |
| ----- | --- | --- | --- | --- | --- |
60
| ssol1/0 60 |     | sedon# 20 |     |     |     |
| ---------- | --- | --------- | --- | --- | --- |
3 ∆
|     |     | 15  |     | 40  |     |
| --- | --- | --- | --- | --- | --- |
40
10
2 20
| 20  |     | 5   |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     |     | 0   |     | 1 0 |     |
-1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0
|        | α   |     | α   |                       | α   |
| ------ | --- | --- | --- | --------------------- | --- |
| Etrain |     |     |     | 5 )%(.smaraporez-non# |     |
25
| 80  |     |     |     | 80  |     |
| --- | --- | --- | --- | --- | --- |
Etest
|     |     | 20     |     | 4   |     |
| --- | --- | ------ | --- | --- | --- |
|     |     | sedon# |     | 60  |     |
ssol1/0 60
15
3 ∆
| 40  |     |     |     | 40  |     |
| --- | --- | --- | --- | --- | --- |
10
2 20
| 20  |     | 5   |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     |     | 0   |     | 1 0 |     |
-1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0
|     | α   |     | α   |     | α   |
| --- | --- | --- | --- | --- | --- |
30
| Etrain |     |     |     | 5 )%(.smaraporez-non# |     |
| ------ | --- | --- | --- | --------------------- | --- |
80
|     |     | 25  |     | 80  |     |
| --- | --- | --- | --- | --- | --- |
Etest
4
| ssol1/0 60 |     | sedon# 20 |     | 60  |     |
| ---------- | --- | --------- | --- | --- | --- |
|            |     | 15        |     | 3 ∆ |     |
40
40
10
2
20
| 20  |     | 5   |     |     |     |
| --- | --- | --- | --- | --- | --- |
1 0
0
-1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0 -1.0 -0.5 0.0 0.5 1.0
|     | α   |     | α   |     | α   |
| --- | --- | --- | --- | --- | --- |
Figure 10: Regularization path over α for different λ (top to bottom {100,10,1}).
18