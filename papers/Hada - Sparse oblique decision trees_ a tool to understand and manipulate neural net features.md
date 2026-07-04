DataMiningandKnowledgeDiscovery(2024)38:2863–2902
https://doi.org/10.1007/s10618-022-00892-7
Sparseobliquedecisiontrees:atooltounderstand
andmanipulateneuralnetfeatures
Suryabhan Singh Hada1 Miguel Á. Carreira-Perpiñán1
· ·
Arman Zharmagambetov1
Received:1April2021/Accepted:5October2022/Publishedonline:9January2023
©TheAuthor(s),underexclusivelicencetoSpringerScience+BusinessMediaLLC,partofSpringerNature2023
Abstract
Thewidespreaddeploymentofdeepnetsinpracticalapplicationshasleadtoagrow-
ing desire to understand how and why such black-box methods perform prediction.
Much work has focused on understanding what part of the input pattern (an image,
say) is responsible for a particular class being predicted, and how the input may be
manipulatedtopredictadifferentclass.Wefocusinsteadonunderstandingwhichof
theinternalfeaturescomputedbytheneuralnetareresponsibleforaparticularclass.
We achieve this by mimicking part of the neural net with an oblique decision tree
havingsparseweightvectorsatthedecisionnodes.UsingtherecentlyproposedTree
Alternating Optimization (TAO) algorithm, we are able to learn trees that are both
highlyaccurateandinterpretable.Suchtreescanfaithfullymimicthepartoftheneu-
ralnettheyreplaced,andhencetheycanprovideinsightsintothedeepnetblackbox.
Further,weshowwecaneasilymanipulatetheneuralnetfeaturesinordertomakethe
netpredict,ornotpredict,agivenclass,thusshowingthatitispossibletocarryout
adversarialattacksatthelevelofthefeatures.Theseinsightsandmanipulationsapply
globally to the entire training and test set, not just at a local (single-instance) level.
WedemonstratethisrobustlyintheMNISTandImageNetdatasetswithLeNet5and
VGGnetworks.
Keywords Decisiontrees Deepneuralnetworks Interpretability
· ·
Responsibleeditor:MartinAtzmueller,JohannesFürnkranz,TomášKliegrandUteSchmid
B SuryabhanSinghHada
shada@ucmerced.edu
B MiguelÁ.Carreira-Perpiñán
mcarreira-perpinan@ucmerced.edu
ArmanZharmagambetov
azharmagambetov@ucmerced.edu
1 UniversityofCalifornia,Merced,USA
123

2864 S.S.Hadaetal.
1 Introduction
Deepneuralnetsareaccurateblack-boxmodels.Theyarehighlysuccessfulinterms
of predictive performance (say, classifying an input image) but remarkably difficult
to understand in terms of how exactly they come up with a prediction for an input.
Bothoftheseissueshavebeenknowntoresearchersandpractitionersformanyyears,
butitisinthe2010sthatdeeplearninghasachievedawild,unexpectedsuccessthat
has attracted widespread attention beyond computer science. In only a few years,
neuralnetshavebecometheworkhorsemodelinanumberofpracticalproblems,in
computervision,speechandlanguageprocessing,games,self-drivingcarsandother
engineeringapplications;legal,financialandmedicalapplications;andmanyothers.
Neural nets now underlie intelligent processing in desktops, cloud computing and
IoT devices. Yet, the way neural nets are defined and optimized, and the sheer size
and complexity of state-of-the-art deep nets, make them very hard to understand in
explanatoryterms.Thisisalsotrueofothermachinelearningmodels,butdeepnets
sitatthefarendintermsofopaqueness.Deepneuralnetsaregenerallynotbasedon
mechanisticmodelsthatinvolvephysicalentitiesinacausalway.Theypurelylearna
correspondencebetweencomplexhigh-dimensionalinputsandoutputsbymeansof
function approximation techniques based on using many adjustable building blocks
(layers, neurons, weights and various nonlinear transformations). Such models can
potentiallyapproximatemanypossiblecorrespondenceswithappropriatechoicesfor
theseparameters,andfindingagoodchoiceispossibleviaanumericaloptimization
algorithmthatminimizesapredictionlossoveralarge,labeleddataset.Theresulting
net can make highly accurate predictions for test inputs, but leaves many questions
unanswered.Wedonotknowwhatagivenneuronorweight,orgroupofthem,codes
for at a level that a human can understand; or what would happen if we remove or
alteragivensetofneuronsorweights;orwhatshouldwechangeintheinputinstance
to change the prediction in a certain way; or what should we change in the trained
nettocorrectawrongpredictioninsomespecificinputinstance.Further,forreasons
notwellunderstood,theclasspredictedbyadeepnetcanbeverysensitivetominute
alterationsoftheinputinwaysthatcanbeusedadversarially.
Most of these questions are not new (Guidotti et al. 2018; Rudin 2019), but they
havebecomeurgentduetothewidespreaddeploymentofdeepnetsinsensitiveappli-
cations. Indeed, this is leading to law changes regarding the use of AI systems and
data, such as the EU General Data Protection Regulation. Erring occasionally in a
movierecommendationsystemisnotserious,butcraftinganadversarialattacksoit
systematically recommends or does not recommend certain movies is. Much more
serious are errors or attacks in legal, financial or medical applications. An example
ofallthesethreeatonceistheprocessingofmedicalinsuranceclaims,anareatradi-
tionallyfraughtwithmoreorlesslegalattemptstoaffectthepaymentoutcomes.The
useofdeepnetsforautomaticclaimdecisionmakingintroducesfurtheropportunities
formanipulationthataremorecreativeanddifficulttodetect(Finlaysonetal.2019).
Moregenerally,thereisaneedtounderstandAIsystemsexperimentally,anddifferent
perspectives(includingbutnotlimitedtocomputerscience)willlikelybenecessary
(Rahwanetal.2019).
123

Sparseobliquedecisiontrees:atooltounderstand… 2865
Ourpaperhastwocontributionsthatcanimproveourabilitytoexplainandmanip-
ulatetraineddeepnets.Firstly,weproposesparseobliquedecisiontreesasatoolto
understand deep nets. Using decision trees is by itself not a new idea. What is new
is the specific, novel type of tree we use, and how we apply it to a given deep net.
Traditionaltreelearningalgorithmstypicallyconstructtreeswhereeachdecisionnode
thresholdsasingleinputfeature(axis-alignedtrees).Althoughsuchtreesareconsid-
ered among the most interpretable machine learning models, this is only true if the
treeisrelativelysmall;itisveryhardtointerpretanaxis-alignedtreewiththousands
ofnodes.Moreimportantly,theaxis-aligneddecisionsareill-suitedtohandledatasets
withmany,correlatedfeatures.Inpractice,axis-alignedtreesusuallyachievetoolow
anaccuracy,andarewhollyinadequateforhigh-dimensionalcomplexinputssuchas
pixelsofanimageorneuralnetfeatures.WecapitalizeonarecentlyproposedTree
AlternatingOptimization(TAO)algorithmwhichcanlearnfarmoreaccuratetreesthat
remainsmallandveryinterpretable,becauseeachdecisionnodeoperatesonasmall,
learnablesubsetoffeatures.
Second,weapplythetreetoaninternallayerofthedeepnet,hencemimickingits
remaining(classifier)layers,ratherthanattemptingtomimictheentiredeepnet.This
allowsustoprobetherelationbetweendeepnetfeaturesandclasses.Asasubproduct,
inspectionofthetreeallowsustoconstructanewkindofadversarialattackswhere
we manipulate the deep net features via a mask to block a specific set of neurons.
Thisgivesussurprisingcontrolonwhatclassthedeepnetwilloutput.Amongother
possibilities,wecanmakeitoutputthesame,desiredclassforalldatasetinstances;
ormakeitneveroutputagivenclass;ormakeitmisclassifycertainpairsofclasses.
Next,wereviewrelatedwork(Sect.2)andtheTAOalgorithm(Sect.3),describe
how we use trees to understand and manipulate deep net features (Sect. 4, 5), and
demonstratethisinMNISTandImageNetdatawithLeNet5andVGG16deepneural
nets(Sect.6).AshortversionofthispaperappearedinHadaetal.(2021).
2 Relatedwork
Thelastfewyearshaveseenmuchworkintheareaofinterpretingorunderstanding,
insomeway,theinternalworkingsofatrainedneuralnetwork.Wedescribethemost
relevantwork,organizedinseveralcategories.
2.1 Featureinversionoractivationmaximization
The idea here is to find what input feature vectors (e.g. what images, for a VGG16
network) produce a certain output under the neural network. This can be done for
individual neurons, with the goal of understanding what “concept” a neuron may
encode,orforanentirelayerofneurons.Mathematically,thisisessentiallyaproblem
ofinvertingtheneuralnetfunction.
One way to do this is to formulate the inversion problem as an optimization: to
minimize the Euclidean distance between the target outputs (ataneuron or layer of
neurons)andtheoutputsgeneratedbytheinputfeaturevectorsought.Initsbasicform,
123

2866 S.S.Hadaetal.
thisideagoesbackdecades(KindermannandLinden1990;Jensenetal.1999),and
hasbeenrevisitedbyvariouspapersrecently.Withimages,naiveapplicationofthis
procedure will generate noisy or unrealistic images. Various approaches have been
proposed to mitigate this, such as regularizing the optimization problem using total
variation(MahendranandVedaldi2016)ordata-drivenpatchpriors(Weietal.2015),
or learning these regularizers by training a deep neural network to generate images
fromthefeatures(DosovitskiyandBrox2016).
Anotherwayistoseekaninputpatternthatwillmaximizetheactivation(output)of
agivenneuron,analogouslytoseekingthe“receptivefield”oftheneuron.Suchpattern
wouldrepresentthepreferredstimulustowhichthatneuronresponds,andmightgive
a clue to what that neuron encodes. This is again an optimization problem over the
inputspaceofthenetwork,whichcanbesolvedinvariousways(e.g.(Simonyanetal.
2014)).Otherworksseektoprovidemultiplepatternsratherthanasingleone,inorder
to find a better characterization of a given neuron (Nguyen et al. 2016, 2017; Hada
etal.2019).Again,thiscanbecombinedwithregularizersorgenerativeadversarial
networkstoobtainrealisticimages.
Finally,therealsoexistapproachesthatarenotbasedonoptimization(Bauetal.
2017;MuandAndreas2020).
2.2 Local,instance-levelexplanations
Thislineofworkseekstoexplaintheneuralnetpredictionforagiveninputinstance
(or group of instances). For example: what part of a given input image was mostly
responsibleforthenetworktoclassifyitasacertainclass?Thisisoftenreferredtoasa
saliencymapoftheimage.Anintuitivewaytodothisisviasensitivityanalysis,suchas
computingorapproximatingthegradientoftheoutputscorewithrespecttotheinput
image(Simonyanetal.2014;ZeilerandFergus2014).WithReLUactivationfunc-
tions,which haveadiscontinuousgradient,thiscan produceartifacts(Sundararajan
etal.2017;Shrikumaretal.2017).Otherapproachesandvariationsexist(Bachetal.
2015;Montavonetal.2016, 2018;Shrikumaretal.2017;Zhouetal.2016;Selvaraju
etal.2017;FongandVedaldi2017;Qietal.2020;Sundararajanetal.2017).Although
saliencymapsarevisuallyappealingandcansometimesagreewithhumanintuition,
itisnotclearhowrobustandconsistenttheyare,andtheycanactuallybemisleading
dependingonthecase(FongandVedaldi2017;Adebayoetal.2018;Ghorbanietal.
2019; Rudin 2019). For example, the saliency map for a given image may be very
similarevenwhenitiscomputedfordifferenttargetclasses.
Another way to seek input features that are particularly important in predicting
a given instance’s class are Shapley values, originally proposed to attribute reward
among players of a cooperative game. Calculating Shapley values is NP-hard, so
theyareapproximatedinpractice(ŠtrumbeljandKononenko2014;Dattaetal.2016;
Merrick and Taly 2020; Lundberg and Lee 2017). The ability of Shapley values to
provideexplanationsthatareusefulforhumanshasalsobeencriticized(Kumaretal.
2020).
Explanation by examples is another way to provide instance-level explanations,
whereweseekwhichinstancesinthetrainingset(onwhichtheneuralnetwastrained)
123

Sparseobliquedecisiontrees:atooltounderstand… 2867
aremostresponsibleforagiveninstancetobeclassifiedasacertainclass.Naively,
this would involve retraining the neural net without each particular instance, to see
whattheeffectofthatinstancewouldhavebeen.Thisiscomputationallyverycostly
and is approximated in various ways (Koh and Liang 2017; Yeh et al. 2018; Pruthi
etal.2020).
Finally,anotherlineofworkforlocalexplanationistoreplacetheneuralnetwork
functionlocallyaroundthegiveninstancewithasimplermodelthatcanbeinterpreted.
One of these methods is LIME (Ribeiro et al. 2016), which describes a somewhat
involved procedure to fit a sparse linear model using a sample of instances near the
giveninstance.Thenonzerocoefficientsinthislinearmodelcanbeusedtogaugethe
importanceofinputfeaturesinthepredictionfortheinstance.Thishasbeenextended
todecisionrules(Ribeiroetal.2018)insteadofalinearmodel.Someoftheselocal
explanation methods can be seen from a common point of view of explaining the
network’soutputasaweightedsumoftheinputfeatures(LundbergandLee2017).
Other works (Singh et al. 2019; Zhang et al. 2019) are based on constructing an
agglomerative clustering tree over the neurons or input features. This is done by
definingasimilaritymeasureforthelatterintermsoftheirabilitytopredictthelocal
instance.Theclusteringtreeprovidesahierarchicalarrangementoftheinputfeatures
andcanbeinspectedbyahumantolookforgroupsofinputfeaturesthatareinfluential
intheoriginalinstance’sprediction.(Althoughtheclusteringtreeiscalleda“decision
tree”in(Zhangetal.2019),itisnotadecisiontreeintheusualsenseofclassification
or regression.) Because of the multiple approximations involved and the lack of a
clearcriterionofwhattheproxymodelissupposedtoexplain,theseapproachesare
somewhatad-hoc.
2.3 Globalexplanationviaaninterpretablemimicoftheneuralnetwork
Thegoalofthesetypesofmethodsistomimictheentiredeepneuralnetviaamore
interpretablemodelsuchasdecisiontreesordecisionrules.Thisthenprovidesaglobal
explanation, applicable to any input instance, unlike the previous, local explanation
methods,whichareonlyvalidnearagiveninstance.
The topic of extracting sets of rules from neural nets was actively researched in
the 1990s (Andrews et al. 1995; McCormick et al. 2013; Guidotti et al. 2018). Two
basicapproacheswereused:inruleextractionassearch(Fu1994;TowellandShavlik
1993), a specialized heuristic search over possible rules was based on the neurons’
connectivity pattern, but this assumed binary activations and did not scale beyond
smallnets.Inruleextractionaslearning,ortheteacher-studentapproach(Cravenand
Shavlik1994,1996;Domingos1998),onetrainsadecisiontreetomimictheneuralnet
byusingthelatter’spredictionsonthetrainingset(possiblyaugmentedwithrandom
instances).Crucially,thesuccessofthisideareliesonthefaithfulnessofthemimic.
Althoughsomeofthesepapers(TowellandShavlik1993;CravenandShavlik1996;
Baesensetal.2003)claimedthattheextractedrulescloselyapproximatetheoriginal
neural net, this was based on very small problems and networks (single-layer). In
suchproblems,traininganaxis-aligneddecisiontreedirectlywasnotfarinaccuracy
123

2868 S.S.Hadaetal.
fromtheneuralnetinthefirstplace,andcouldproducearelativelysmalltreeanda
correspondinglysmallsetofrules.
Thefundamentalproblemwiththisisthattraditionalalgorithmstolearndecision
treesordecisionrules,suchasCART(Breimanetal.1984)orC4.5(Quinlan1993),
based on axis-aligned trees, are unable to learn accurate enough trees to be useful
mimicsofaneuralnetexceptinverysmall,low-dimensionalproblems.Theyfallfar
shortofhandlingthelarge,deepneuralnetsthatareusedincurrentcomputervision
applications,forexample.
2.4 Ourworkincontext
Our work belongs to the category of global explanation via an interpretable mimic
thatisadecisiontree,withtwoimportantdifferences.First,weuseaspecialtypeof
decision tree, a sparse oblique tree. This can be trained to be much more accurate
thanaxis-aligned,CART-typetrees,whichmakesitmorelikelythatonecanobtaina
faithfulmimic.Atthesametime,thesparseobliquetreeremainsinterpretable.Thisis
becausethetreesizeisfarsmallerthanthatofanaxis-alignedtree(indepthandnumber
of nodes), and because it uses relatively few features in each decision node. As we
showinourexperiments,inspection(manualorautomatic)ofthenonzeroweightsin
thedecisionnodesleadstoimportantinsightsaboutthetree(andabouttheneuralnet),
andshowshowtomanipulatetheneuralnetfeaturestoaltertheclassificationresult
inadesiredway.(Furtherintermsofinterpretability,counterfactualexplanationscan
besolvedexactlyandefficientlyforobliquetrees(Carreira-PerpiñánandHada2021;
Hadaetal.2021),althoughwedonotusethishere.)
A second difference is that we do not aim at replacing the entire neural net, but
at replacing the classifier portion of a deep net (globally over all instances). This
allows us to study the relation between the deep net features (neuron activations
at a certain internal layer) and the output classes—note that those features were
specifically learnedby theneuralnetduringtrainingwiththegoal ofpredicting the
classes optimally. This is unlike much of the work cited earlier, which studies the
relationbetweeninputfeatures(e.g.pixels)andneuronactivationsatacertainlayer.
(In Zharmagambetov et al. (2021), we also train a sparse oblique tree on features
obtained frommultiple pretraineddeep neuralnetworks,butthe goal thereisnotto
interpretaneuralnet,buttoachieveamodelwithhigheraccuracythanthepretrained
networks.)
3 LearningsparseobliquetreeswiththeTreeAlternating
Optimization(TAO)algorithm
We briefly explain the Tree Alternating Optimization (TAO) algorithm; the original
referencesgivemoredetails(Carreira-Perpiñán2022;Carreira-PerpiñánandTavallali
2018).Amongothertypesoftrees,TAOcanlearnsparseobliquetrees.Thesehave
a constant label at each leaf and a linear decision function at each decision node
but, crucially, the decision function only uses a typically small, learned subset of
123

| Sparseobliquedecisiontrees:atooltounderstand… |     |     |     | 2869 |
| --------------------------------------------- | --- | --- | --- | ---- |
features(seeFig.5).TAOachievesthisbyoptimizinganobjectivefunctionwhichis
thesumoftheclassificationlossandan penalty(withhyperparameterλ 0)on
|     |     | 1   |     |    |
| --- | --- | --- | --- | --- |
theweightvectorsofthedecisionnodes(similartoaLASSO(Hastieetal.2015)but
on each decision node). Each TAO iterationdecreases this objective and consists of
optimizing over groups of non-descendant nodes (such as all the nodes at the same
depth),wheretheobjectivecanbeshowntoseparateoverthenodesinthegroup.The
optimization over each node can be shown to be equivalent to a simpler, “reduced”
problemtakingtheformofamajorityclassifierataleaf,andabinaryclassifierata
decisionnode.Inthelatter,eachinstancereachingthenodeisassigneda“pseudolabel”
indicatingthechildthatgivesthelowestlossunderthecurrenttree.TAOusesafixed
tree structure while iterating, which is automatically pruned because subtrees may
eventuallyreceivetraininginstancesofthesameclass,ornotreceiveinstancesatall;
thisisheavilypromotedbythe penaltyonthedecisionnodes,whichcandriveall
1
weightsinadecisionnodetozero,thusmakingitredundant.
Inmoredetail,consideratreeT ofagivenstructure(usually,completeofdepthΔ)
|                 | andparametersΘ | θ   | .Foradecisionnodei,θ |            |
| --------------- | -------------- | --- | -------------------- | ---------- |
| withnodesinaset |                | i   | i                    | i consists |
|                 | N              | = { | } l∈eNafi,θ          |            |
oftheweightsandb iasofthehyperplane .Fo ra 1,...,K isaclasslabel.
|     |     |     | i ∈ { } |     |
| --- | --- | --- | ------- | --- |
Now,totrain T onadataset x ,y N D 1,.. ., K ofinp utinstancesand
n n }n 1 R
theirlabels,weoptimizethefollowing=objectivefunction: { ⊂ ×{ }
N
|     | minE(Θ) | (y ,T(x | Θ)) λ φ (θ ) | (1) |
| --- | ------- | ------- | ------------ | --- |
|     |         | n n     | i i          |     |
|     | Θ =     | L ;     | +            |     |
|     |         | n 1     | i            |     |
|     |         | =      | ∈N          |     |
where (.,.)isthecross-entropyloss,φ (θ )isaregularizationtermwithhyperpa-
i i
L
rameter λ 0 (we will use  1 regularization on the weight vectors of the decision

nodes).TAOreliesontwotheorems:aseparabilitycondition,andareducedproblem
overeachnode(seedetailsandproofsinCarreira-Perpiñán(2022),Carreira-Perpiñán
andTavallali(2018)).Here,wedescribethembriefly.
Theorem31 (separabilitycondition).Consideranypairofnodesi and j.Assumethe
parametersofallothernodes(Θ )arefixed.Ifnodesi and j arenotdescendants
rest
| ofeachother,then | E(Θ)canberewrittenas: |             |                 |     |
| ---------------- | --------------------- | ----------- | --------------- | --- |
|                  | E(Θ)                  | E(θ i ) E(θ | j ) E(Θ rest ). | (2) |
|                  |                       | = +         | +               |     |
Inotherwords,theseparabilityconditionstatesthatanysetofnon-descendantnodesof
atreecanbeoptimizedindependently.NotethatE(Θ )canbetreatedasaconstant
rest
| sincewefixΘ | .   |     |     |     |
| ----------- | --- | --- | --- | --- |
rest
(reducedproblem).Forasinglenodei,optimizingoveritsparameters
Theorem32
simplifies to a well-defined reduced problem over the instances that currently reach
| nodei (thereducedset |     | 1,...,N ). |     |     |
| -------------------- | --- | ---------- | --- | --- |
i
|     | R ⊂{ | }   |     |     |
| --- | ---- | --- | --- | --- |
– For a decision node i, the reduced problem can be written as a 0/1 loss binary
classificationproblem,wherethetargetclassreferstotheleftorrightsubtree.For
this we assign a “pseudolabel” (y left,right ) indicating the child that
|     |     | ∈ { | }   |     |
| --- | --- | --- | --- | --- |
123

2870 S.S.Hadaetal.
givesthelowestlossunderthecurrenttree.Thus,thereducedproblemtakesthe
form:
E(θ ) L (y ,(f(x ) θ )) λφ (θ ) (3)
i n n i i i i
= ; +
n ∈R i
where L n isthe0/1lossand f i R D left,right isthedecisionfunction
: →{ }
at node i with parameters θ (which, for an oblique tree, is defined by a linear
i
classifier). This is an NP-hard problem, but it can be approximated by using a
convexsurrogateloss.Inthiswork,weuse regularizedlogisticregression(φ
1 i
=
)andoptimizeitusingLIBLINEAR(Fanetal.2008).
1
·
– For a leaf node i, the reduced problem can be shown to have an exact solution,
givenbysettingtheleaflabelθ tothemajorityclassamongalltheinstancesin
i
itsreducedset .
i
R
Theabovetheoremsmeanthatwecanmonotonicallyreducetheobjectivefunction(1)
bycyclingoverthenodesinthetreeinsomeorderandoptimizingeachnode’sparam-
etersbysolvingitsreducedproblem.Wecanoptimizeanysubsetofnon-descendant
nodesinparallel,e.g.allnodesatthesamedepth.Asforthetreestructure(andinitial
parameters),thiswilldependonthedataset,butwefindthatusingacompletetreeof
largeenoughdepthΔandrandom(Gaussian0,1)initialparametersoftenworkswell.
In a sense, TAO operates similarly to how a neural net (or other machine learn-
ing models) aretrained:by fixing themodelstructureandoptimizing adesiredloss
functioniteratively.However,insteadofusinggradients(whicharenotavailablefora
decisiontree),itusesalternatingoptimization.Modelselectionoverthetreestructure
happensautomaticallybyusingalargeenoughtreestructureandlettingTAOprune
it via the  penalty (as has also been done to prune weights and neurons in neural
1
nets,e.g.(Carreira-PerpiñánandIdelbayev2018)).Wedescribetheentiretreetrain-
ingprocessinFig.1.AsdescribedinCarreira-Perpiñán(2022);Carreira-Perpiñánand
Tavallali(2018),thecomputationalcomplexityofoneTAOiterationisupperbounded
bythedepthofthetreetimesthecostofsolvingalogisticregressionproblemonthe
entiredataset.Thisisbecausethereducedsetsofallthenodesatthesamedepthform
apartitionoftheentiretrainingset.
Wecancontrolthetradeoffbetweenaccuracyandsparsity(intermsofthesizeof
thetreeandthenumberofnonzeroweightsatthenodes),andhencecontroltheamount
of feature selection and interpretability, similarly as with the LASSO regularization
path(Hastieetal.2015).Wesimplytraceafamilyoftreesofdecreasingaccuracyand
increasingsparsityasthehyperparameterλgoesfrom0to .
∞
TAO considerably improves (Zharmagambetov et al. 2021) over the traditional,
widelyusedalgorithms(suchasCART(Breimanetal.1984)orC4.5(Quinlan1993))
that are based on greedy recursive partitioning based on a purity measure. This is
becausethesealgorithmshavenousefulguaranteesconcerningtheclassificationloss
to start with, and are only moderately effective with axis-aligned trees. The same
approachcanbeusedtotrainobliquetrees(Breimanetal.1984;Murthyetal.1994),
butitresultsinlarge,nonsparseandhighlysuboptimaltreesthatoftendonotimprove
overaxis-alignedones.Thesearethefundamentalreasonswhyaxis-alignedtreesare
123

Sparseobliquedecisiontrees:atooltounderstand… 2871
inputtrainingset
{
(xn,yn)
}
N
n=1⊂
RD
×{
1,...,K
}
initialtreeT
repeat
fori nodesofT,visitedinreverseBFS
∈
if iisaleafthen
θi← majority-classlabelinthereducedset Ri
else
generatepseudolabelsy n foreachinstancen ∈Ri
θi← minimizerofthereducedproblem(eq.(3))
untilstop
postprocessT:removedeadbranches&puresubtrees
returnT
Fig.1 Pseudocodeforthetreealternatingoptimization(TAO)algorithm(Carreira-Perpiñán2022;Carreira-
PerpiñánandTavallali2018).Visitingeachnodeinreversebreadth-firstsearch(BFS)ordermeansscanning
depthsfromdepth(T)downto1,andateachdepthprocessing(inparallel,ifsodesired)allnodesatthat
depth.“stop”occurswheneithertheobjectivefunctiondecreaseslessthanasettoleranceorthenumberof
iterationsreachesasetlimit
theonlytypeoftreethatiswidespreadinpractice,atleastuntilnow.TAOalsoimproves
treeensembles(forests)considerably:ifusingTAOinsteadofaCART-typealgorithm
(asdoneinrandomforests(Breiman2001)andXGBoost(ChenandGuestrin2016)),
the resulting forest contains fewer, shallower trees but is consistently more accu-
rate,whetherusingbaggingorboostingtoensemblethetrees(Carreira-Perpiñánand
Zharmagambetov2020;Zharmagambetovetal.2020, 2021),Zharmagambetovetal.
(2021).Finally,TAOcanalsobeusedtotrainnovelformsoftreemodels(Zharmagam-
betovetal.2021),Zharmagambetovetal.(2021).
4 Sparseobliquetrees:amicroscopetoobserveadeepneuralnet
Our overall approach is as follows (see Fig. 2). Assume we have a trained deep net
y f(x)forclassificationofaninputinstancex R D intoK classes,soyisavector
= ∈
of K softmax values encoding (an approximation to) the class distribution given x.
Wewritethenetf(x) g(F(x))asthecompositionofafeature-extractionlayerF,so
z F(x) R F arethe = deepnetfeatures(neuronoutputsatthatlayer),andaclassifier
lay = er y ∈ g(z) consisting of the rest of the net.1 This includes as particular cases
=
of features the raw inputs x (where F is the identity) and the class label or softmax
outputs(wheregistheidentity),butwewillusuallybemoreinterestedinfeaturesat
anintermediatelayer.Eachneuronatthatlayercanbeconsideredasafeaturedetector
whichencodessomepropertyorconceptoftheinputpatternx,whichmaybeuseful
(incombinationwithotherneurons’concepts)toprovideinformationfororagainst
oneormoreclasses.
1 Thisisanoperationaldefinition,sincetheoriginalnetwastrainedwithoutanexplicitconstructionof
featureextractionandclassifier,andindeedsuchdistinctionisblurredinsomearchitecturessuchasResNets
(Heetal.2016).Otherarchitectures,suchasLeNet5(LeCunetal.1998)andVGG(SimonyanandZisserman
2015)dohaveaclearseparationintofeatureextraction(basedonconvolutional,poolingandsubsampling
layers),andclassification(fully-connectedMLP).
123

| 2872 |     |     |     |     |     | S.S.Hadaetal. |
| ---- | --- | --- | --- | --- | --- | ------------- |
y=f(x),
entireneuralnet
y
K output
classes
y=g(z),
|     |     | z=F(x), |     | classifierpart |     |     |
| --- | --- | ------- | --- | -------------- | --- | --- |
x
| Dinput |     | F neuralnetfeatures |     | (mimicked |     |     |
| ------ | --- | ------------------- | --- | --------- | --- | --- |
bytree)
features
Fig.2 Mimickingpartofaneuralnetwithadecisiontree.Thefigureshowstheneuralnety f(x)
= =
g(F(x)),consideredasthecompositionofafeatureextractionpartz F(x)andaclassifierparty g(z).
|     |     |     |     | =   |     | =   |
| --- | --- | --- | --- | --- | --- | --- |
Forexample,fortheLeNet5neuralnetof(LeCunetal.1998)inthediagram,thiscorrespondstothefirst
4layers(convolutionalandsubsampling)followedbythelast2,fully-connectedlayers,respectively.The
“neuralnetfeature”vectorzconsistsoftheactivations(outputs)ofF neurons,andcanbeconsideredas
featuresextractedbytheneuralnetfromtheoriginalfeaturesx(pixelvalues,forLeNet5).Weuseasparse
obliquetreetomimictheclassifierparty g(z),bytrainingthetreeusingasinputtheneuralnetfeatures
=
zandasoutputthecorrespondingground-truthlabels
Assumewehaveadataset(usuallytheoneusedtotrainthenet)
| (x ,y ) | N R | D 1,...,K | ofinputinstancesandtheirlabels.Then: |     |     |     |
| ------- | --- | --------- | ------------------------------------ | --- | --- | --- |
| n n }n  | 1   |           |                                      |     |     |     |
| {       | = ⊂ | ×{        | }                                    |     |     |     |
N
1. Trainasparseobliquetreey T(z)withTAOonthetrainingset (F(x n ),y n )
|     |     | =   |     |     | {   | }n 1 ⊂ |
| --- | --- | --- | --- | --- | --- | ------ |
F 1,...,K . Explore the interpretability-accuracy tradeoff over a usef=ul a
R
| ×{  |     | }   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
rangeofthesparsityhyperparameterλ 0, )andpickafinaltree.Usuallythis
∈[ ∞
willbeatreewithclosetohighestvalidationaccuracyandassparseaspossible.
2. Inspectthetreetofindinterestingpatternsaboutthedeepnet.
Ourgoalistoachieveatreethatbothmimicswellthedeepnetandisassimpleas
possible.Weachievethisbytrainingthetreeonthesametrainingsetasthenet(using
thelatter’sfeaturesbuttheground-truthlabels2).
Step 2 is purposely vague. There is probably a wealth of information in the tree
regardingthefeatures’meaningandeffectontheclassification,bothatthelevelofa
specificinputinstanceormoreglobally.Inthispaperwefocusononespecificpattern
describedinthenextsection.
5 Manipulatingthefeaturesofadeepnettoalteritsclassification
behavior
Ouroverallobjectiveistomanipulatethevalueofthedeepnetfeaturesz R F toalter
inacontrolledwaytheclasspredictedbythenet.Wewillnotaltertheweightsofthenet, ∈
| i.e.,Fandgremainthesame.Wejustalterzintoamaskedz |     |     |     |     | μ(z) μ | z μ     |
| ------------------------------------------------ | --- | --- | --- | --- | ------ | ------- |
|                                                  |     |     |     | =   | =      | ×  + + |
2
Thisisequivalenttousingthedeepnetpredictionsaslabels(teacher-studentapproach),becauseourdeep
netsachievenearlyzerotrainingerror.
123

Sparseobliquedecisiontrees:atooltounderstand… 2873
F,
via a multiplicative and an additive mask μ × ,μ + R respectively (where “ ”
|     |     | ∈   |    |
| --- | --- | --- | --- |
meanselementwisemultiplication).Specifically,wehave:
| Originalnet:y | f(x) | g(F(x)) | (4) |
| ------------- | ---- | ------- | --- |
= =
| Originalfeatures:z | F(x) |     | (5) |
| ------------------ | ---- | --- | --- |
=
| Maskednet:y | f(x) | g(μ(F(x))) | (6) |
| ----------- | ---- | ---------- | --- |
= =
| Maskedfeatures:z | μ(F(x)) | μ(z). | (7) |
| ---------------- | ------- | ----- | --- |
|                  | =       | =     |     |
WedemonstratethemaskingoperationinFig.3.Inthesimplest,mostintuitiveversion
ofthemask,wejustneedabinarymultiplicativemaskz μ zwhereμ 0,1 F.
|     |     | × × |      |
| --- | --- | --- | ---- |
|     |     | =  | ∈{ } |
Usinganadditivemaskandreal-valuedmasksmakesthemanipulation’seffectmore
robustandhardertodetect.
Wewillconstructamaskbyinspectingthetree,specificallybyobservingtheweight
ofeachfeatureineachdecisionnodeofthetree.Byselectivelyzeroingsomefeatureswe
canguaranteethatanyinstancewillfollowaspecificchildinagivennodeandhence
directinstancesasdesiredtowardsatargetleaf.Undersomeassumptions,wewillbe
abletoguaranteeadesiredeffectifusingthetree,i.e.,intheclassifiery T(μ(F(x))).
=
Then we will apply the mask to the deep net as y g(μ(F(x))). While we cannot
=
guaranteeanythinginthemaskednet,wecanreasonablyexpectsimilarresultsifthe
tree is a good mimic of the classifier g, and indeed our experiments show that the
maskednetbehaveslikethemaskedtreemostofthetimes.
Atadecisionnodei,thedecisionruleis“ifwTx
b 0thengotorightchild,
|     | i   | + i  |     |
| --- | --- | ----- | --- |
elsego toleftchild”,where w F isthe weightvector and b Rthebias.We
| i R |     | i   |     |
| --- | --- | --- | --- |
∈ ∈
willassumethefollowing(throughoutweuseelementwisenotationasneeded,asin
“z 0”):

– The deep net features are nonnegative: z F(x) 0. This is true for ReLUs,
|     | =   |    |     |
| --- | --- | --- | --- |
whichareusedinmostdeepnetsatpresent.
– Thebiasateachdecisionnodei ofthetreeiszero:b 0.Thisholdsverywell
i
=
inthetreeswetrained,specifically b i w i ateachdecisionnodei.
|     | | |  |     |     |
| --- | ------- | --- | --- |
Iftheseassumptionsdonothold,itisstillpossibletodesignmasksthatworkreliably
insomecases.Wementionsomeofthembutgenerallyleavesuchdetailsoutofthis
paper.
5.1 Divertingallinstancestoonechild
Wegiveabasicprocedure“μ ,μ Node-Mask(i,c)”thatunderliesourmask
| × + |     |     |     |
| --- | --- | --- | --- |
←
construction.Assumeaninstancezreachesadecisionnodeiinthetree.Node-Mask
producesamaskthatguaranteesthatz μ z μ goesleft(right)ifc left
|     | = ×  | + + | =   |
| --- | ----- | --- | --- |
(right)child,foranyinstancezinthetrainingset.Essentially,Node-Maskallowsus
to“cut”asubtree,sobyapplyingitasneededwecancutsubtreestoleaveonlyacertain
pathinthetreeforanyinstancetofollowandhenceeffectadesiredclassification.
Node-Maskworksasfollows.Calltheweightvectorwandbiasb(assumedzero
anyway)atnodei.Writethemw.l.o.g.asw (w w w )andz (z z z )where
|     | = 0 | = 0 |     |
| --- | --- | --- | --- |
w 0,w < 0andw > 0containthe zero,neg − ativ + eandpo sitivew − eig + htsinw,
0
= − +
123

2874 S.S.Hadaetal.
Originalnetwork
z
z1
z2
⎡ ⎤
z3
⎢ . ⎥
⎢ . . ⎥
⎢ ⎥
F ⎢z ⎥ g
x
−−−→ −−−→
⎢ ⎢
⎢
k z−
k
1⎥ ⎥
⎥−−→ −−→
y
⎢z ⎥
⎢ k+1⎥
⎢ . ⎥
⎢ . ⎥
⎢ . ⎥
⎢ ⎥
⎢z ⎥
⎢ ⎢ F z− 1⎥ ⎥
⎢ F ⎥
⎣ ⎦
Masked network
z μ z
z1 1 z1
z2 1 z2
⎡ ⎤ ⎡ ⎤ ⎡ ⎤
z3 0 0
⎢ . ⎥ ⎢.⎥ ⎢ . ⎥
⎢ ⎢ . . ⎥ ⎥ ⎢ ⎢ . .⎥ ⎥ ⎢ ⎢ . . ⎥ ⎥
F ⎢z ⎥ ⎢0⎥ ⎢ 0 ⎥ g
x −−−→ −−−→ ⎢ ⎢ ⎢ k z− k 1⎥ ⎥ ⎥ ⎢ ⎢ ⎢ 1 ⎥ ⎥ ⎥ =⎢ ⎢ ⎢ z k ⎥ ⎥ ⎥−−−−−→ −−−−−→ y
⎢z ⎥ ⎢0⎥ ⎢ 0 ⎥
⎢ k+1⎥ ⎢ ⎥ ⎢ ⎥
⎢ . ⎥ ⎢.⎥ ⎢ . ⎥
⎢ . ⎥ ⎢.⎥ ⎢ . ⎥
⎢ . ⎥ ⎢.⎥ ⎢ . ⎥
⎢ ⎥ ⎢ ⎥ ⎢ ⎥
⎢z ⎥ ⎢0⎥ ⎢ 0 ⎥
⎢ ⎢ F z− 1⎥ ⎥ ⎢ ⎢1 ⎥ ⎥ ⎢ ⎢z ⎥ ⎥
⎢ F ⎥ ⎢ ⎥ ⎢ F⎥
⎣ ⎦ ⎣ ⎦ ⎣ ⎦
Fig.3 Toporiginalnetwork.Bottommaskingoperationinthenetwork.Thesymbols’meaningisasfollows:
inputx,featureextractionpartofthenetworkF,originalfeaturesz,binarymaskcreatedusingthetreeμ,
modifiedfeaturesz,classifierpartofthenetworkg,originaloutputy,andmodifiedoutputy
andz 0isreorderedaccordingtothat.Call , and thecorrespondingsets
0
ofindi  cesinw.ThenwTz b wTz wTS z S w − ithwTS z + 0andwT z 0.
So if z 0 then wTz + b = 0 an−d − z + wou+ld + go to the−rig − ht  child and if + z +  0
thenwT−
z
=
b 0andz
+
woul

dgototheleftchild.Hence,ourmasksareasfo
+
llo
=
ws:
togoleft, + μ  0,1 F isabinaryvectorcontainingonesat ,zerosat and
×
(meaninganyv ∈ al { ue)a } t ;andμ 0isavectorcontaining S s − mallpositi S ve + value ∗ s
0 +
S 
at andzeroelsewhere.Togoright,exchange“ ”and“ ”intheprocedure.The
add S i − tivemaskμ isnecessaryonlyifthefeaturesi − n all + happentobezero(which
+
would produce wTz b 0 and be on the decision S− boundary). This is unlikely to
+ =
happenunless containsveryfewfeatures,butstilltheadditivemaskisusefulto
S−
123

Sparseobliquedecisiontrees:atooltounderstand… 2875
push wTz away from the boundary and hence make it more likely that the masked
deepnetwillperformasdesired3.
5.2 Masks
We now show how to construct masks that effect a certain class outcome. For each
case,westatethedesiredgoalandthecorrespondingmask.Inthemanipulationsbelow
wemayuseNode-Maskrepeatedlyoverseveralnodestoconstructthemask(which
isappliedtothefeaturevectorandhenceappliesgloballytoeachnode).Inthatcase,
wewillonlyusethemultiplicativemaskproducedbyNode-Maskateachnode,and
createtheadditivemaskattheendgiventhefinalmultiplicativemask.
– All class k to class k : let k k 1,...,K . For any instance x
1 2 1 2
= ∈ { }
originallyclassifiedask ,classifyitask .Foranyotherinstance,donotalterits
1 2
classification.Thiscaseonlyworksiftheclassesk andk areleafsiblings(have
1 2
the sameparent).Classk mayberepresented by multipleleaves sinceweonly
2
needtodealwithoneofthem(thesiblingofk ).Mask:simplyapplyNode-Mask
1
totheparentoftheleavesofk andk .
1 2
– None to classk:letk 1,...,K .Foranyinstancexoriginallyclassifiedas
∈{ }
k,classifyitasanyotherclass.Foranyotherinstance,donotalteritsclassification.
Mask:simplyapplyNode-Masktotheparentofeachleafofk andcombinethe
resultingmultiplicativemasksasextended-AND(definedbelow).Finally,addthe
additivemask.
Strictly speaking, we can guarantee that class-k instances are classified as some
otherclass,butnotthatwedonotaltertheclassificationofotherinstances.Thisis
becausethefeaturesthataremaskedoutmayappearinothernodesandpossibly
affectthepathofaninstance.However,withourdeepnetsthenumberoffeatures
masked out is very small and the mask works well. If the features selected in a
nodeonlyappearinthatnode,theintheireffectispurelylocal,ofcourse.
– All to classk:let k 1,...,K .Classifyallinstancesxasclassk.Mask:
∈ { }
findthepathfromtheroottotheleafofclassk.Ateachnodei inthepath,apply
Node-Mask(todivertinstancesalongthepath)andkeepthemultiplicativemask
only.Thefinalmultiplicativemask,elementwise,hasa0whereanyofthenode
masks has a 0, a 1 where all node masks have no 0s but at least one 1, and
∗
elsewhere.Thismasksoutallthe“undesired”featuresthatmightdivertusfromthe
path.Equivalently,thisisthelogicalextended-ANDofallthemultiplicativemasks
alongthepath(whereweextendANDtomeanAND( ,0) 0,AND( ,1) 1
∗ = ∗ =
andAND( , ) ).
∗ ∗ =∗
Thiswouldnotworkifthemultiplicativemaskiszeroatallfeatures,butthisis
unlikelyifthenodeshavesparseweightvectors,ashappensinourexperiments.
This also works if class k is represented as multiple leaves. We simply take the
unionofthemasksovereachleaf.
3 Infact,justsettingz μ (i.e.,replacingthefeatureswiththeadditivemaskwithoutevenusingthe
= +
multiplicativemask)wouldworkinthetree.Thisboilsdowntoreplacinganyincomingfeaturevectorwith
afixedfeaturevectorofknownclassificationunderg.Butthismakestheattackveryobvious:thesoftmax
valuesoutputtedbythenetarethesameforeveryinstance.
123

2876 S.S.Hadaetal.
– None to a subset of classes:wecanapplythisinsomecasesdependingon
thetree.ItsimplyfollowsbyapplyingNode-Masktoagivendecisionnode.The
classesinthesubtreethatiscutoutcannotbereachedbyanyinstance.Applying
this to multiple nodes and creating the multiplicative mask by extended-AND
removesthecorrespondingsubtreesandtheirclasses.
5.3 Hidingtheadversarialattack
Ourmotivationformanipulatingtheneuralnetfeatureswastoillustratehowthesparse
oblique trees are able to gain information about how the network works internally.
However, such manipulations can also be seen as adversarial attacks at the feature
level (which may or may not be practically feasible). Applying the attack is very
simple,asitneedsnooptimization(whichisthecasewithmanypixel-levelattacks).
Wecanalsomaketheattacklessobvious.Inpracticewithourtrees(whichhave
sparse weight vectors), the above masks only require setting to 0 or 1, always in
the same place, a small number of features ( 10–40) out of the total (hundreds or
≈
thousands), so they would be easily detectable to an observer of either the masked
featuresorthesoftmaxvaluesattheoutput.Wecaneasilyrandomizetheabovemasks
(andmakethemcontinuousratherthanbinary)sothattheystillworkasintendedbut
vary for each instance. First, the wildcard indices in the multiplicative masks can
∗
take any real value (positive, negative or zero). Second, the additive mask can take
anypositivevaluesaslongastheyaresmallenough.Third,theretypicallyisasubset
offeatureswhicharenotusedinanynodeofthetree,sotheycanalsotakeanyreal
value.
Changingsomeadditionalfeaturesmayofcoursehavesomeunintendedeffectsin
thedeepnet.However,thiscanbereducedbyapplyingtheaboverandomizationtoa
small,itselfrandomnumberoffeaturesinthemask.Also,sincethedeepnetfeatures
aretheoutputofaReLU,someofthemare0tostartwith,sothemaskhasnoeffect
thereanyway.
Some of our manipulations are less detectable than others by their nature. Two
classes that are siblings (such as ‘4’ and ‘9’ in the MNIST tree; Fig. 11) are likely
more similar than if they are far apart in the tree, so misclassifying them (as one of
ourmasksdoes)wouldnotraisemuchsuspicion.Theattackdescribedinthissection
isawhite-boxattack,whichmeanstheattackerhasaccesstothemodel.
6 Experiments
Wehaveevaluatedourtreesandmasksthoroughlyontwodeepnets:
– VGG16 (Simonyan and Zisserman 2015) in a subset of 16 classes of ImageNet
(Deng et al. 2009), for which we select the F 8192 neurons from its last
=
convolutionallayer.Table2givesthenetworkarchitecture.
– LeNet5inMNISTon10digitclasses(LeCunetal.1998),forwhichweselectthe
F 800neuronsatlayerconv2asfeatures.Table3givesthenetworkarchitecture.
=
123

| Sparseobliquedecisiontrees:atooltounderstand… |     | 2877 |
| --------------------------------------------- | --- | ---- |
Table1 Classesinour
|     | Labelid | Class |
| --- | ------- | ----- |
ImageNetsubsetandtheirid(for
| referenceinotherfigures) | 0   | Goldfish      |
| ------------------------ | --- | ------------- |
|                          | 1   | Baldeagle     |
|                          | 2   | Goose         |
|                          | 3   | Killerwhale   |
|                          | 4   | Siberianhusky |
|                          | 5   | Whitewolf     |
|                          | 6   | Tigercat      |
|                          | 7   | Lion          |
|                          | 8   | Airliner      |
|                          | 9   | Containership |
|                          | A   | Fireengine    |
|                          | B   | Schoolbus     |
|                          | C   | Speedboat     |
|                          | D   | Sportscar     |
|                          | E   | Warplane      |
|                          | F   | Coralreef     |
Forbothofthem,wecantraintreesthataccuratelymimicthedeepnetclassifierg.The
treesgiveremarkableinsightintherelationofdeepnetfeaturestoclassesandallow
ustoconstructmasksthatindeedworkasintendedinthedeepnetformostinstances.
Wedescribethisindetailnext.
6.1 ResultsonVGG16(subsetofImageNetdataset)
We selected a subset of 16 classes from the ImageNet object classification dataset
(Dengetal.2009);theyarelistedinTable1.Foreachclasswesplittheavailableimages
into200fortest,100forvalidationand1000fortraining(total20800images).We
usedaVGG16deepnet(SimonyanandZisserman2015)withthearchitectureshown
in Table 2, which takes as input color images of 64 64 pixels. We fine-tuned a
×
pretrainedVGG16forourImageNetsubsetof16classes.Wetrainthenetworkusing
Nesterov’sacceleratedgradientmethodfor20epochsusingminibatches ofsize32,
learningrate0.02andmomentumrate0.9.OurresultingVGG16netachievesanerror
of0.2%(training)and6.79%(test).Weselectthe F 8192neuronsfromVGG16’s
=
lastconvolutionallayerasfeaturesonwhichtotrainthetree.
We trained sparse oblique trees on these features with the TAO algorithm, using
our own Python implementation of TAO. We used as initial tree structure for TAO
a complete tree of depth 6 (total 127 nodes), which we found sufficient to produce
smallabutaccuratetreesinthiscase,andrandominitialvaluesfortheweightsatthe
nodes.Thedecisionnodesarehyperplanesandeachleafcontainsasingleclasslabel.
Weconstructedacollectionoftreesoverarangeofthesparsityhyperparameterλ
∈
0, )(theregularizationpath).WeranTAOfor40iterations(whenitapproximately
[ ∞
123

2878 S.S.Hadaetal.
102
101
100
10-1
-2 -1 0 1 2 3 4 5
c
rorre
eert
detceleS
net(test)
tree(train)
tree(test)
net(train)
50
40
30
20
10
0
-2 -1 0 1 2 3 4 5
c
sedon
6
5
4
3
2
1
0
eert
detceleS
shtped
# nodes
depth
12
10
8
6
4
2
0
-2 -1 0 1 2 3 4
c
esraps
eert
detceleS
Fig.4 Classificationerror(trainingandtest)andnumberofnodesandofnonzeroweightsofthetreesasa
functionofλforVGG16.Theverticallineindicatesthetreeweselectedasmimic(λ 1)
=
123

Sparseobliquedecisiontrees:atooltounderstand… 2879
converged) to learn each tree. Figure 4 shows, for each tree, its error (training and
test),size(depthandnumberofnodes)andnumberofnonzeroweightsasafunction
ofthesparsityhyperparameterλ 0, ).
∈[ ∞
Thetreewiththelowesterroroverthevaluesofλweconsideredoccurredforλ
=
0.01.Ithasdepth6and51nodes,andanerroror0%(training)and7.62%(test).Ituses
only4423featuresoutofthetotal8192.Wedidnotusethistree,insteadweselected
asmimicthetreeforλ 1,whichisquitesmaller(depth6andonly39nodes)but
=
hasnearlythesameerror(0%training,7.90%test).Itusesjust1366features(17%
ofthetotal8192).ItserrorisveryclosetothatofVGG16,soweexpectthetreeto
beagoodmimicofthenet.Wenormalizethefinaltreesoeachnodeweightvector
hasnorm1.ThistreeisshowninFig.5.Wealsodiscussanothertreethatisslightly
lessaccuratebutwhichhasexactlyoneleafperclassandisevenmoreinterpretable
(Fig.6).Thistree(λ 33)hasanerrorof1.79%(training)and9.56%(test);ithas
=
31nodesandusesjust408features(5%ofthetotal8192).
6.1.1 Inspectingthesparseobliquetrees
Figure4showsthatasweincreaseλandthereforeimposeincreasingsparsityonthe
tree(intermsofbothtreesizeandnumberofnonzeroweightsinthedecisionnodes),
thetrainingerrorincreasessteadilybutthetesterrorremainsaboutconstant,soboth
curvesapproach,meetforsomeλvalueandapproximatelycoincidefromthatpoint
on.WefindthisbehaviorinbothMNISTandImageNet.Thisprovidesopportunities
tofindatreewithprettygoodaccuracybutsignificantlysparse.
Figure 5 shows the tree we use as mimic. Its training and test errors are close to
those of VGG16, so we expect it to be a good mimic, which indeed happens (see
masks later). The top of the figure shows the class histogram at each node, i.e., the
distributionofclassesonthesubsetoftraininginstancesthatanodereceives.These
histogramsshowhowthetreehierarchicallysplitsclassesverycrisply;indeed,ithas
only 20 leaves for 16 classes. In the bottom of the figure, the weight vector at each
decisionnodeshowsthatveryfewfeaturesareusedateachnode;indeed,83%ofthe
features are not used at any node, so their values are irrelevant for classification in
thetree.Thisalsoholdsnearlyperfectlyforthedeepnet,thatis,thefeatureselection
insights obtained for the tree transfer to the neural net. It suggests that some of the
featuresandhenceneuronsandweightsofthenetarepracticallyredundant,orperhaps
thattheycodeforpropertiesthatareusefulforonlyafewspecificinstances.Thisis
notsurprisingifonenotesthatdeepnets(atleast,aspresentlydesigned)seemtobe
vastlyoverparameterizedandcanbesignificantlycompressedbypruningweightsand
neurons(Carreira-PerpiñánandIdelbayev2018).
Figure 6 shows a very interesting tree, obtained for a larger λ value so that there
isexactlyoneleafperclass(thesmallestnumberofleavespossibleunlessweignore
classes). This tree has very few nonzero weights yet its test error is reasonable, so
it probably extracts features that robustly classify most images. Also, its structure
remainsunchangedforawiderangeofλ.Inspectingitshowsanintuitivehierarchy
ofclassesthatseemprimarilyrelatedtothebackgroundorsurroundingsofthemain
objectintheimage.Itsleftmostsubtree{warplane,airliner,schoolbus,fireengine,
sportscar}consistsofman-madeobjectsoftenfoundonroads.However,{container
123

| 2880 |     |     |     |     |     | S.S.Hadaetal. |     |
| ---- | --- | --- | --- | --- | --- | ------------- | --- |
16000
|     |     | 6531 |     |     | 9469 |     |     |
| --- | --- | ---- | --- | --- | ---- | --- | --- |
0123456789ABCDEF
|     | 5000         |     | 1531        | 5020         |     | 4449 |      |
| --- | ------------ | --- | ----------- | ------------ | --- | ---- | ---- |
|     | 4000 1000(D) |     | 1204 327(6) | 4020 1000(7) |     | 2000 | 2449 |
2000 2000 1000(0) 204(F) 1673 2347 1000(9) 1000(C) 1796 653(3)
1000(E1)000(81)000(B1)000(A) 1603 70(6) 13471000(4) 1000(1)796(F)
6031(060)0(5) 3471(030)0(2)
1( b=0.040886)
|     |                | 2( b=-0.080825) |         |                | 3( b=0) |         |     |
| --- | -------------- | --------------- | ------- | -------------- | ------- | ------- | --- |
|     |                |                 |  1      |  0 -1          |         |         |     |
|     | 4( b=-0.13669) |                 | 5( b=0) | 6( b=-0.52553) |         | 7( b=0) |     |
8( b=0) 9(D) 10( b=0) 11(6) 12( b=0) 13(7) 14( b=0) 15( b=-0.035122)
16( b=0) 17( b=0) p13 20(0) 21(F) p6 24( b=0) 25( b=0) p7 28(9) 29(C) 30( b=0) 31(3)
|     |     | p0  | p15 |     | p9  | p12 | p3  |
| --- | --- | --- | --- | --- | --- | --- | --- |
32(E)33(8)34(B)35(A) 48( b=0)4950(6( )b=0.1039541)(4) 60(1)61(F)
| p14 | p8 p11 p10 |     |     | p61 p4 |     | p1  | p15 |
| --- | ---------- | --- | --- | ------ | --- | --- | --- |
96(967)(5) 1001(031)(2)
p62p5
p3 p2
Fig.5 TreeselectedasmimicforVGG16features(λ 1),withatrainingerrorof0%andatesterrorof
=
7.90%.Top:classhistograms;weshowthenumberoftraininginstancesreachingthenodeand,forleaves,
theirlabel.Bottom:weightvectorateachdecisionnodeandanimagefromtheirclassateachleaf;we
showthenodeindex,bias(alwayszero)and,forleaves,theirlabel.Weplottheweightvector,ofdimension
8192,asa91 91square(thelastpixelsareunused),withfeaturesintheoriginalorderinVGG16(which
isdeterminedduringtrainingandarbitrary,hencetherandomaspectoftheimages),andcoloredaccording ×
totheirsignandmagnitude(positive,negativeandzerovaluesareblue,redandwhite,respectively).You
mayneedtozoomintheplot
ship, speedboat} (man-made objects found on the sea) appears in the rightmost
subtree, together with {killer whale, bald eagle, coral reef}, all of which are also
typically found on the sea or on the air. Yet {goldfish} appears in a single subtree
quiteseparatefromallotherclasses:indeed,thisfishisfoundonfishbowls(notthe
sea)inthetrainingimages.Asubtreeinthemiddlecontainsanimalsinlandnatural
environments (forest, snow, grass, etc.): {tigercat,whitewolf,goose,Siberianhusky,lion}.
Andsoon.Thisisconsistentwithpreviousworksthathavefoundthat,insomespecific
cases,thereasonwhyadeepnetclassifiesanobjectasacertainclassiscausedbythe
background or more generally by some confounding variables (Ribeiro et al. 2016;
Zechetal.2018).Itpointstoapossiblevulnerabilityofthenet,inthatitmaybadly
123

| Sparseobliquedecisiontrees:atooltounderstand… |     |     |     |     |     |     | 2881 |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- |
16000
6010 9990
0123456789ABCDEF
|      | 5016        | 994(0) |      | 4975        |         | 5015   |              |
| ---- | ----------- | ------ | ---- | ----------- | ------- | ------ | ------------ |
|      | 4031 985(D) |        |      | 3997 978(7) |         | 1989   | 3026         |
| 2008 | 2023        |        | 2013 | 1984        | 1003(9) | 986(C) | 2019 1007(3) |
1007(E)1001(8)1006(B)1017(A) 998(6)1015(5)989(2)995(4) 1003(1)1016(F)
1( b=0)
|         | 2( b=-1.0653) |      |          |                | 3( b=0)  |         |          |
| ------- | ------------- | ---- | -------- | -------------- | -------- | ------- | -------- |
|         |               |      | -1  0  1 |                |          |         |          |
|         | 4( b=0)       | 5(0) |          | 6( b=-0.60836) |          | 7( b=0) |          |
| 8( b=0) | 9(D)          | p0   |          | 12( b=0) 13(7) | 14( b=0) |         | 15( b=0) |
16( b=0) 17( b=0) p13 24( b=0) 25( b=0) p7 28(9) 29(C) 30( b=0) 31(3)
|             |            |     |       |                   | p9  | p12   | p3    |
| ----------- | ---------- | --- | ----- | ----------------- | --- | ----- | ----- |
| 32(E) 33(8) | 34(B)35(A) |     | 48(6) | 49(5) 50(2) 51(4) |     | 60(1) | 61(F) |
| p14 p8      | p11 p10    |     | p6    | p5 p2 p4          |     | p1    | p15   |
Fig.6 LikeFig.5butforλ 33.Thistreehasexactlyoneleafperclass(total16classes),andaslightly
=
highererror(trainingerror1.79%,testerror9.56%)
misclassify an object that happens to appear in an unusual background (say, a bald
eaglestandingonaroad).
6.1.2 Manipulatingthedeepnetfeaturesviamasks
Wederivemasksusingthemimictree(λ 1).Figures7,8showconfusionmatrices,
=
whichareself-explanatory,overallinstances(testandtraining,respectively)regarding
thedeepnet,thetreeandthemaskeddeepnet.Generally,themasksaffectthedeep
netclassificationinthesamewayasthetree.Thisistobeexpectedsincethetreehas
a very similar error and confusion matrix as the net, but it is still surprising in how
well it works in most cases. This also indicates that certain neurons of the deep net
(thosecriticallyinvolvedinthemasks)playawell-definedroleintheclassification.
Thenumberoffeaturesthatamaskcriticallyneedstoperformitsjobisverysmall,
around200(outof8192);forMNISTitismuchsmaller,around40.
Weemphasizetwothings.First,theseconfusionmatricesareconstructedusingall
instances(trainingortest).Hence,ourconclusionsarerobustandglobal,unlikeother
works that either work locally on a single instance by design, or work globally but
123

| 2882 |                                |                              |                      |              |                              | S.S.Hadaetal. |
| ---- | ------------------------------ | ---------------------------- | -------------------- | ------------ | ---------------------------- | ------------- |
|      |                                | groundtruthvsdeepnet         |                      |              | featuresselected             |               |
|      |                                |                              | vstree               |              | bythetree                    |               |
|      |                                | 0123456789                   | 0123456789           |              | 0123456789                   | 1             |
|      |                                | hturt dnuorG                 | ten lanigirO         |              | ten lanigirO                 |               |
|      |                                | ABC                          | ABC                  |              | ABC                          |               |
|      |                                | D                            | D                    |              | D                            |               |
|      |                                | E F                          | E F                  |              | E F                          |               |
|      |                                | 01234O5rig6in7a8l n9eAtBCDEF | 0123456T7re8e9ABCDEF |              | 01234M5od6if7ied8 n9eAtBCDEF | 0             |
|      | ....................Allclassk1 |                              |                      | toclassk2    | ....................         |               |
|      | 8                              | 14 14 8                      | 10 11                | 11           | 10 9                         | 12 12 9       |
|      | 0123456789 →                   | 0123456789 →                 | 0123456789 →         | 0123456789 → | 0123456789 →                 | 0123456789 →  |
|      | ten lanigirO                   | ten lanigirO                 | ten lanigirO         | ten lanigirO | ten lanigirO                 | ten lanigirO  |
|      | ABC                            | ABC                          | ABC                  | ABC          | ABC                          | ABC           |
|      | D E                            | D E                          | D E                  | D E          | D E                          | D E           |
F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF
...................................Nonetoclassk ...................................
| k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 k=7 |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC  | ABC | ABC | ABC | ABC  | ABC | ABC ABC |
| ---- | --- | --- | --- | ---- | --- | ------- |
| D EF | D E | D E | D E | D EF | D E | D E D E |
01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF
| k=8 | k=9 | k=A | k=B | k=C | k=D | k=D k=E |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC | ABC | ABC | ABC | ABC | ABC | ABC ABC |
| --- | --- | --- | --- | --- | --- | ------- |
| D   | D   | D   | D   | D   | D   | D D     |
| EF  | E F | E F | E F | EF  | E F | E F E F |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
....................................Alltoclassk ....................................
| k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 k=7 |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC | ABC | ABC | ABC | ABC | ABC | ABC ABC |
| --- | --- | --- | --- | --- | --- | ------- |
| D   | D   | D   | D   | D   | D   | D D     |
| EF  | E F | E F | E F | EF  | E F | E F EF  |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
| k=8 | k=9 | k=A | k=B | k=C | k=D | k=D k=E |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC  | ABC  | ABC   | ABC   | ABC  | ABC   | ABC ABC    |
| ---- | ---- | ----- | ----- | ---- | ----- | ---------- |
| D EF | D EF | D E F | D E F | D EF | D E F | D E F D EF |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
Fig.7 ConfusionmatricesforVGG(testset).Firstrowleft:ground-truthvsdeepnet,anddeepnetvstree.
Firstrowright:deepnetvsdeepnetwithonlythefeaturesselectedbythetree.Secondrow:All class
k1to classk2(selectedexamples).Thirdandfourthrow:None to classk.Fifthandsixthrow:All
to classk
showgoodresultsonahandfulofinstancesonly.Byreportingtheresultsaggregated
overallinstances,wedemonstratetherobustnessofourmasks.Second,whatweshow
istheresultofapplyingtheoriginalVGG16deepnettothemaskedfeatures,notof
applyingthetreetothemaskedfeatures(whichwouldworkperfectlybyconstruction).
Thisshowsthatthemasksconstructedusingthetreemimictransferalmostperfectly
tothedeepnet.
Letusanalyzethedifferentpanelsofthefigureinmoredetail:
– Thetwoconfusionmatrices“groundtruthvsdeepnetvstree”(whichlooklikea
diagonal yellow line) show that the deep net and the tree predictions are almost
123

Sparseobliquedecisiontrees:atooltounderstand… 2883
|     |                                | groundtruthvsdeepnet         |                      |              | featuresselected             |              |
| --- | ------------------------------ | ---------------------------- | -------------------- | ------------ | ---------------------------- | ------------ |
|     |                                |                              | vstree               |              | bythetree                    |              |
|     |                                | 0123456789                   | 0123456789           |              | 0123456789                   | 1            |
|     |                                | hturt dnuorG                 | ten lanigirO         |              | ten lanigirO                 |              |
|     |                                | ABC                          | ABC                  |              | ABC                          |              |
|     |                                | D                            | D                    |              | D                            |              |
|     |                                | EF                           | EF                   |              | EF                           |              |
|     |                                | 01234O5rig6in7a8l n9eAtBCDEF | 0123456T7re8e9ABCDEF |              | 01234M5od6if7ied8 n9eAtBCDEF | 0            |
|     | ....................Allclassk1 |                              |                      | toclassk2    | ....................         |              |
|     | 8                              | 14 14 8                      | 10 11                | 11           | 10 9                         | 12 12 9      |
|     | 0123456789 →                   | 0123456789 →                 | 0123456789 →         | 0123456789 → | 0123456789 →                 | 0123456789 → |
|     | ten lanigirO                   | ten lanigirO                 | ten lanigirO         | ten lanigirO | ten lanigirO                 | ten lanigirO |
|     | ABC                            | ABC                          | ABC                  | ABC          | ABC                          | ABC          |
|     | D EF                           | D EF                         | D EF                 | D E F        | D EF                         | D E F        |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
...................................Nonetoclassk ...................................
| k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 k=7 |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC  | ABC  | ABC  | ABC  | ABC | ABC  | ABC ABC  |
| ---- | ---- | ---- | ---- | --- | ---- | -------- |
| D EF | D EF | D EF | D EF | D E | D EF | D E D EF |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF F 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
| k=8 | k=9 | k=A | k=B | k=C | k=D | k=D k=E |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC  | ABC | ABC  | ABC  | ABC | ABC  | ABC ABC |
| ---- | --- | ---- | ---- | --- | ---- | ------- |
| D EF | D   | D EF | D EF | D E | D EF | D D EF  |
|      | EF  |      |      | F   |      | EF      |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
....................................Alltoclassk ....................................
| k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 k=7 |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC | ABC | ABC | ABC | ABC | ABC | ABC ABC |
| --- | --- | --- | --- | --- | --- | ------- |
| D   | D   | D   | D   | D   | D   | D D     |
| EF  | EF  | EF  | EF  | E F | EF  | E F EF  |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
| k=8 | k=9 | k=A | k=B | k=C | k=D | k=D k=E |
| --- | --- | --- | --- | --- | --- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC  | ABC  | ABC   | ABC   | ABC   | ABC   | ABC ABC   |
| ---- | ---- | ----- | ----- | ----- | ----- | --------- |
| D EF | D EF | D E F | D E F | D E F | D E F | D EF D EF |
01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF 01234M5od6if7ied8 n9eAtBCDEF
Fig.8 LikeFig.7butforthetrainingset
identical to each other and to the ground truth, indicating that the tree is a good
mimic(blueis0andyellow1).
– Theconfusionmatrix“featuresselectedbythetree”(whichlookslikeadiagonal
yellowline)showsthatifwemaskoutallfeaturesinVGG16exceptthoseselected
bythetree,theclassificationabilityremainsthesame,indicatingthatthetreeis
abletodetectasubsetoffeaturesthatareenoughforgoodclassification.
– The6confusionmatrices“All classk to classk ”demonstrate(inselected
|     |     |     |     | 1   | 2   |     |
| --- | --- | --- | --- | --- | --- | --- |
combinations of k 1 and k 2 ) that this mask works quite reliably. Ideally, entry
(k ,k ) should become 0 and entry (k ,k ) should become 1, with the rest of
|     | 1 1 |     |     | 2 1 |     |     |
| --- | --- | --- | --- | --- | --- | --- |
the entries remaining unchanged. We often observe a bluish vertical bar at k ,
2
indicating that a small proportion of the k instances are classified as a class
1
| differentfromk |     | .   |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | --- |
2
123

2884 S.S.Hadaetal.
– The 16 confusion matrices “None to class k” (for k 0,1,...,F ) should
∈ { }
ideally make the entry at (k,k) equal to zero (this entry is originally equal to
(almost) 1) and distribute its mass over any other classes (entries (k,k ) with

k k ),withtherestoftheentriesremainingunchanged.Thissucceedsbetterin

=
someclassesthaninothers,butgenerallyworkswell.
– The 16 confusion matrices “All to class k” (for k 0,1,...,F ) should
∈ { }
ideallylooklikeaverticalyellowlineatk,andindeedtheydoinallcases.
Alloftheaboveistrueforbothtrainingandtestinstances,althoughthemaskswork
slightlybetterforthetraininginstances.
6.1.3 Illustrationofthemaskswithanactualimage
Figure 9 illustrates the mask behavior in an image not in the dataset. The middle
columnhistogramsshowthedeepnetfeatures(groupedbyclass).Ineachrow,thetop
histogramshowsthefeaturevalues,andthebottomhistogramshowsthenumberof
featuresselectedforeachclass.Next,weshowhowmaskingthefeaturesdrastically
altersinacontrolledwaythesoftmaxoutput.Inrow2,whenweapplytheAll to
class “Siberian husky”mask,thenetworknowclassifiestheimageas“siberian
husky”.Similarly,inrow6,whenweapplytheAll to class “bald Eagle”mask,
thenetworknowclassifiestheoriginalimageas“baldeagle”withlargeconfidence,
comparedtorow1,wherewithoutthemaskthesoftmaxvaluefor“baldeagle”isclose
tozero.Wealsoshowhowthemaskcorrelateswithsuperpixels(perceptualgroupsof
pixelsobtainedbyoversegmentation)intheimage,eithermanuallycropped(row3)
oroptimizedtoinvertthedesireddeepnetfeatures(row4).
Toobtainresultslikethoseabove,thegeneralprocedureisasfollows.Firstly(inan
offlinephase),wetrainthetreemimicandconstructasubsetoffeatures foreach
k
S
classk,usingtheAll to classkmask.Thisdefinesascoreforaninputimagexas
s (x) F(x),where F(x)isthefeaturei computedbythedeepneuralnet
fo k rx.W = eca i n∈St k hen i discardthetr i eeandtheclassifierpartofthedeepnet.Allweneed
∑
isthefeature-extractionpartofthedeepnetandtheclasssets ,..., .
1 K
S S
Then (in an online phase), given an input image and a target class k, we split
the image into superpixels, compute the score for each superpixel, and report the
superpixels with lowest score (most salient). We use the quick-shift segmentation
algorithm(VedaldiandSoatto2008)implementedintheskimage4librarytogenerate
superpixels.Wesetallparameterstotheirdefaultvaluesexceptthecolor-spaceand
image-spaceproximityratio,andthecut-offpointfordatadistances,whichwesetto
0.2(moreweighttoimage-spaceproximity)and200(fewerclusters),respectively.
6.1.4 Controlexperiments
We run some experiments to test the robustness of our findings (see details in
Appendix2):
1. We trained trees with TAO using 5 different random initial trees and verified we
can achieve very similar results to those we report (masks, etc.). Note that, for a
4 https://scikit-image.org/docs/0.15.x/
123

| Sparseobliquedecisiontrees:atooltounderstand… |     |     |     | 2885 |
| --------------------------------------------- | --- | --- | --- | ---- |
1
|          | 300               | outside axis limit |                   |     |
| -------- | ----------------- | ------------------ | ----------------- | --- |
|          | tnuoc erutaef 200 |                    | 0.8               |     |
| lanigirO |                   |                    |                   | l13 |
|          | 100               |                    | eulav xamtfos 0.6 |     |
0
|     | 0.27               |                   | 0.4              |              |
| --- | ------------------ | ----------------- | ---------------- | ------------ |
|     | eulav erutaef 0.18 |                   | l4               |              |
|     | 0.09               |                   | 0.2              |              |
|     |                    |                   | 00 l 1l 2l 3 l 5 | l1 0l1 1l1 2 |
|     | 0.00 0 1 2 3 4     | 5 6 7 8 9ABCDEFun | 1 2 3 4 5 6789   | A B C DEF    |
ecapserutaef
|        |                   | outside axis limit | 1 l4 |     |
| ------ | ----------------- | ------------------ | ---- | --- |
| niksaM | tnuoc erutaef 200 |                    |      |     |
|        | 150               |                    | 0.8  |     |
Alltoclass 100
|                 | 50                 |     | eulav xamtfos 0.6 |     |
| --------------- | ------------------ | --- | ----------------- | --- |
| “Siberianhusky” | 0                  |     |                   |     |
| maskisapplied   | 1.26               |     | 0.4               |     |
|                 | eulav erutaef 1.22 |     | 0.2               |     |
1.17
| −−−−−−−−−−−−→ | 1.12      |                   | 00 l 1l 2l 3 l 5 | l1 0l1 1l1 2l 13 |
| ------------- | --------- | ----------------- | ---------------- | ---------------- |
|               | 0 1 2 3 4 | 5 6 7 8 9ABCDEFun | 1 2 3 4 5 6789   | A B C D EF       |
niksamlaunaM
ecapsegami
|     |     | outside axis limit | 1 l4 |     |
| --- | --- | ------------------ | ---- | --- |
tnuoc erutaef 300
|     | 200  |     | 0.8           |     |
| --- | ---- | --- | ------------- | --- |
|     | 100  |     | eulav xamtfos |     |
|     | 0    |     | 0.6           |     |
|     | 0.34 |     | 0.4           |     |
eulav erutaef 0.23
0.2
0.11
|                           | 0.00              |                    | 00 l 1 1l 2 2l 3 3 4 l 5 5 6789 | l1 A 0l1 B 1l1 C 2l D 13 EF |
| ------------------------- | ----------------- | ------------------ | ------------------------------- | --------------------------- |
|                           | 0 1 2 3 4         | 5 6 7 8 9ABCDEFun  |                                 |                             |
| egaminiksaM deniatboecaps |                   |                    | 1 l4                            |                             |
| serutaefyb                | 300               | outside axis limit |                                 |                             |
|                           | tnuoc erutaef 200 |                    | 0.8                             |                             |
|                           | 100               |                    | eulav xamtfos 0.6               |                             |
0
|     | 0.30 |     | 0.4 |     |
| --- | ---- | --- | --- | --- |
eulav erutaef 0.20
|     | 0.10           |                   | 0.2               |                  |
| --- | -------------- | ----------------- | ----------------- | ---------------- |
|     |                |                   | l 1l 2l 3 l 5     | l1 0l1 1l1 2l 13 |
|     | 0.00 0 1 2 3 4 | 5 6 7 8 9ABCDEFun | 00 1 2 3 4 5 6789 | A B C D EF       |
ecapserutaef
outside axis limit
| niksaM | tnuoc erutaef 150 |     | 1 l1 |     |
| ------ | ----------------- | --- | ---- | --- |
|        | 100               |     | 0.8  |     |
Alltoclass 50
|               | 0             |     | eulav xamtfos 0.6 |     |
| ------------- | ------------- | --- | ----------------- | --- |
| “Baldeagle”   | 1.37          |     |                   |     |
|               | eulav erutaef |     | 0.4               |     |
| maskisapplied | 1.35          |     |                   |     |
|               | 1.34          |     | 0.2               |     |
−−−−−−−−−−−−→ 1.32 0 1 2 3 4 5 6 7 8 9ABCDEFun 0 l2l3l4l5 l10l11l12l13
|     | class-based feature group |     | 0123456789ABCDEF |     |
| --- | ------------------------- | --- | ---------------- | --- |
classes
Fig.9 Illustrationofmasksforaparticularimage(VGG16networkonImageNetsubset).Column1shows
theimagemasks(whenavailable).Column2summarizesthe8192featurevaluesastwohistograms:onthe
upperpanel,thenumberoffeaturesineachclassgroup(listedintheXaxisas0–F,where“ ”meansfeatures
∗
notusedbythetree);onthelowerpanels,theaveragefeaturevalue(neuronactivation)perclassgroup.
Column3showsthehistogramofcorrespondingsoftmaxvalues.Row1showstheoriginalimage.Row2
showsamaskinfeaturespacetoclassifyitas“Siberianhusky”.Row3showsamaskmanuallycropped
intheimage,whosefeaturesresemblethoseofrow2.Row4showsamaskinfeaturespaceobtainedby
findingthetop-3superpixelswhosefeaturesmostresemblethoseofthemaskedfeaturesofrow2.Row5
showsamaskinfeaturespacetoclassifytheimageas“baldeagle”
givenneuralnetworkanddataset,itisconceivablethatwecouldlearnverydifferent
trees(evenresultingindifferentmasks)becauseoflocaloptimainthetreetraining,
correlationorredundancyoffeaturesintheneuralnetwork,etc.Butregardlessof
that,because we evaluate themasksnotjustinthetreebutintheoriginal neural
network,wecanclaimthatthechosenmasksworkasdescribedearlier.
2. WetrainedaCARTtree(Breimanetal.1984;Therneauetal.2019;Pedregosaetal.
2011)ontheVGG16features.Thiscreatedahugeaxis-alignedtreewitherrorof
1.97%(training)and21.2%(test),depth54,1381nodesandusing619features.
123

2886 S.S.Hadaetal.
ThelargetesterrormeansthetreeisinadequateasamimicofVGG16,andthetree
sizewouldmakeituselessforexplanationpurposesanyway.Thisdemonstratesthe
inabilityofCARTtolearnaccuratetreesforcomplex,high-dimensionaldata.
3. WetrainedaTAOtreeonarotatedversionoftheVGG16features,i.e.,multiplying
the feature vector by an orthogonal matrix. This has the effect of mixing all the
features in an invertible way. Without sparsity (λ 0), the TAO tree achieves
=
identicalerrortotheunrotatedfeatures’tree(sincetheobliquenodescanabsorb
anylineartransformation).However,whenincreasingλinordertoforcethetreeto
usefewfeatures,thetesterrorjumpsto11.3%,muchhigherthanwiththeunrotated
features.ThisshowsthatthefeatureslearntbyVGGarespecialinthattheyseem
tooperateinsmallgroupsassociatedwithclasses,ratherthanallormostfeatures
participatingineachclass.
6.2 ResultsonLeNet5(MNISTdataset)
WetrainedsparseobliquetreesonfeaturesobtainedbytheLeNet5neuralnetarchi-
tectureforMNIST.TheresultsagreequalitativelywiththoseforVGG16:weareable
toobtaintreeswithnearlythesameerrorastheoriginalnet,hencegoodmimics;the
masksweconstructworkasdesiredinnearlyallthetrainingandtestinstances;and
thetreeishighlyinterpretable.
Specifically, we train a LeNet5 net on the 60000 training images for MNIST, of
28 28pixels(10handwrittendigitclasses),andreporttesterrorsonthe10000test
×
instances(LeCunetal.1998).TheLeNet5architectureisinTable3.Wetrainthenet-
workusingNesterov’sacceleratedgradientmethodfor100epochsusingminibatches
ofsize512,learningrate0.02(updatedevery20epochswithafactorof0.992)and
momentumrate0.9.Thetrainingerroris0.00545%andthetesterroris0.61%.We
selectthe F 800neuronsatlayerconv2asfeaturesonwhichtotrainthetree.
=
WeusedTAOwithaninitialtreestructureofdepth5(total63nodes)andrandom
initial values for the weights at the nodes. We constructed acollection of trees over
arangeofthesparsityhyperparameterλ 0, )(theregularizationpath),running
∈ [ ∞
TAOfor40iterations(whenitapproximatelyconverged)tolearneachtree.Figure10
shows, as a function of λ, the error (training and test), number of nodes and depth
of each tree, and number of nonzero weights in each tree (total over all its nodes).
AswiththeVGG16trees,asweincreaseλandthereforeimposeincreasingsparsity
onthetree(intermsofbothtreesizeandnumberofnonzeroweightsinthedecision
nodes),thetrainingerrorincreasessteadilybutthetesterrorremainsaboutconstant,
so both curves approach, meet for some λ value and approximately coincide from
thatpointon.Thisprovidesopportunitiestofindatreewithprettygoodaccuracybut
significantlysparse.Asweincreaseλ,thetreesizedecreases(numberofnodesand
depth)andthenumberofnonzeroweightsdecreases.
Weselectedasmimicthetreeforλ 20,withdepth5andonly27nodes.Ithasan
=
errorof1.28%(training)and1.67%(test),whichisveryclosetothatofLeNet5,so
weexpectthetreetobeagoodmimicofthenet.Thebestclassifiertree(forλ 5)
=
hadadepthof5and27nodes,andanerroror0.59%(training)and1.46%(test).Itis
123

Sparseobliquedecisiontrees:atooltounderstand… 2887
possibletoreducethiserrorevenmorebyusingalargertree,buttheoneweobtained
isgoodenoughasamimicandtoobtainmasks.
Figure11showsthetreeselectedasmimic.Theclasshistogramsateachnodeshow
how the tree hierarchically splits classes very crisply; indeed, it has only 14 leaves
for 10 classes (only digit classes ‘2’, ‘6’, ‘7’, ‘9’ appear with 2 leaves each). The
blurry average image at each leaf shows significant shape variability, indicating the
featureshavesuccessfullylearnedtoignoresuchwithin-classvariability.Theweight
vectorateachdecisionnodeshowsthatveryfewfeaturesareusedateachnode;295
features(37%ofthetotal800)arenotusedatanynode,sotheirvaluesareirrelevant
forclassificationinthetree.Thisalsoholdsnearlyperfectlyforthedeepnet.
Figures 12, 13 show the confusion matrices for our different masks, which work
nearlyperfectlyinthetraininginstancesandonlyslightlylesssointhetestones.The
numberoffeaturesweneedtomaskoutineachcaseisverysmall,around40(outof
800features).Somemasksworkmorereliablythanothers.Classifyingallinstances
asclasskworkssurprisinglywellnomatterthechoiceofk.Misclassifyingclassk as
1
k (wherek musthaveasingleleafwhichisasiblingofk )worksalsowell,although
2 1 2
afewinstancesfromotherclassesaresometimesclassifiedask .Notclassifyingany
2
instance as class k works also well but fails with some instances, which remain as
classk.
InisinterestingtocompareourtreewithaTAOtreetraineddirectlyonthepixel
values (hence not associated with any deep net), such as that in (Fig. 1 Carreira-
PerpiñánandTavallali(2018)).ThetreetrainedonLeNet5featuresismuchsmaller,
sparserandaccurate.Thebesttesterrorforatreeonpixels(panel2of(Fig.1Carreira-
PerpiñánandTavallali(2018))is5.69%foratreeofdepth8and75nodes(inourown
experimentswecangetthisdowntoaround5%).Thiserrorisremarkablylowfora
treebut,comparedtoourtree,theerrorismuchbigger,andthetreeisquitelargerand
messier.Asmaller,moreinterpretabletreeonpixelvaluesshowninpanel3of(Fig.1
Carreira-PerpiñánandTavallali(2018))achievesanerrorofaround10%(trainingor
test)withdepth7and33nodes.Sincethetreeoperatesontherawpixels,inspection
oftheweightvectorsisveryinformativeinidentifying“strokes”thatcharacterizethe
differencebetweenadigit4andadigit9,forexample.Astoapplyingourmasksatthe
pixellevel,thisisfarlesssuccessfulifwewantguaranteesfor(nearly)allinstances;
however,itmaybepossibletomakethisworkforaspecificimage.
7 Discussionandlimitationsofourwork
Decisiontreesareagoodchoiceofinterpretableclassifierbecausetheyhandlemultiple
classesnaturally,dofeatureselectionautomatically,haveahierarchicalstructurethat
promotes an increasing specialization from the root towards the leaves, and can be
inspected. Sparse oblique trees improve this by using few features at each node, so
they explicitly show the influence of groups of features on classes. This makes it
possibletofindimportantsubsetsoffeaturesefficientlyamongthousandsofpossible
features. For such trees to be useful, it is critical to be able to train them to high
accuracysotheycanmimic(partof)adeepnet,whichtheTAOalgorithmdoes.
123

2888 S.S.Hadaetal.
102
101
100
10-1
10-2
0 1 2 3 4 5
c
rorre
eert
detceleS
net(test)
tree(train)
tree(test)
net(train)
30
25
20
15
10
5
0
0 1 2 3 4 5
c
sedon
5
4
3
2
1
0
eert
detceleS
shtped
# nodes
depth
35
30
25
20
15
10
5
0
0 1 2 3 4 5
c
esraps
eert
detceleS
Fig.10 Classificationerror(trainingandtest)andnumberofnodesandofnonzeroweightsofthetreesas
afunctionofλforLeNet5.Theverticallineindicatesthetreeweselectedasmimic(λ 20).Compare
=
thiswithFig.4
123

Sparseobliquedecisiontrees:atooltounderstand… 2889
55000
18194 0123456789 36806
16221 1973 4971(5) 31835
7938 8283 204 1769(7) 10667 21168
2635(6) 5303(2) 5481(0) 2802(6) 37(9) 167(2) 5282(4) 5385(9) 10175 10993
6205(13)970(75)626(35)367(8)
1( b=0)
2( b=0) 3( b=0)
1 0 -1
4( b=0) 5( b=0) 6(5) 7( b=0)
8( b=0) 9( b=0) 10( b=0) 11(7) 14( b=0) 15( b=0)
16(6) 17(2) 18(0) 19(6) 20(9) 21(2) 28(4) 29(9) 30( b=0) 31( b=0)
60(1)61(7)62(3)63(8)
Fig.11 TreeselectedasmimicforLeNet5features(λ 20).Top:classhistograms;weshowthenumberof
=
traininginstancesreachingthenodeand,forleaves,theirlabel.Bottom:weightvectorateachdecisionnode
andaverageoftraininginstancesateachleaf;weshowthenodeindex,bias(alwayszero)and,forleaves,
theirlabel.Weplottheweightvector,ofdimension800,asa29 29square(thelastpixelsareunused),
×
withfeaturesintheoriginalorderinLeNet5(whichisdeterminedduringtrainingandarbitrary,hencethe
randomaspectoftheimages),andcoloredaccordingtotheirsignandmagnitude(positive,negativeand
zerovaluesareblue,redandwhite,respectively).Youmayneedtozoomintheplot
As shown by our masks, deep net features participate in a coordinated way in
predictingaclass,wheresmallgroupsofspecificfeaturesencodeinformationspecific
for some decisions (rather than, say, all features participating in all classes for all
instances).Thatwefindfeaturesspecializedforclassesisnotthatsurprising—some
featuresmustprovideinformationforsomeclasses,afterall.Whatissurprisingisthe
smallsizeofthesegroupsoffeatures.Thisisprobablypartlyduetotheneuralnetwork
beingheavilyoverparameterized.Whatdothesegroupsofneuralnetfeaturesrepresent
anyway? Unfortunately, in spite of the high activity in this area (as summarized in
Sect.2),atpresenttheresearchcommunitydoesnothaveasystematicunderstanding
ofwhat“concepts”theseneuralnetfeaturesmaybeencoding;elucidatingthisremains
anopenresearchproblem.
Our findings are remarkably similar to recent findings in visual neuroscience
(Marsheletal.2019;Carrillo-Reidetal.2019)thatshowthatverysmallgroupsofneu-
123

2890 S.S.Hadaetal.
groundtruthvs featuresselected .......Allclassk1 toclassk2 .......
deepnetvstree bythetree 4 9 9 4 3 8 8 3 1 7
→ → → → → 0123456789
0123456789
Original net
hturt dnuorG
0123456789
0123456789
Tree
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
1
0
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
..................................Nonetoclassk ..................................
k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7 k=8 k=9
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
...................................Alltoclassk ...................................
k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7 k=8 k=9
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
0123456789
0123456789
Modified net
ten lanigirO
Fig.12 ConfusionmatricesforLeNet5(testset).Topleft:ground-truthvsdeepnet,anddeepnetvstree.
Topmiddle:deepnetvsdeepnetwithonlythefeaturesselectedbythetree.Topright:All classk1to
classk2(selectedexamples).Middle:None to classk.Bottom:All to classk
groundtruthvs featuresselected .......Allclassk1 toclassk2 .......
deepnetvstree bythetree 4 9 9 4 3 8 8 3 1 7
→ → → → → 0123456789
012O3rigi4nal5 ne6t789
hturt
dnuorG
0123456789
0123T4ree56789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
1
0
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
..................................Nonetoclassk ..................................
k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7 k=8 k=9
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
...................................Alltoclassk ...................................
k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7 k=8 k=9
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
0123456789
012M3odi4fied5 ne6t789
ten
lanigirO
Fig.13 LikeFig.12butforthetrainingset
rons(around20)inmouseprimaryvisualcortexseemtocodeforspecificproperties
orbehaviors.Infact,removingallvisualinputtothemouseanddirectlystimulating
thoseneuronstriggersthesamebehaviors—analogouslytowhatourmasksdo.
Recentwork(KuhnandKacker2019)triestoaddresstheproblemofidentifyinga
groupofnfeatures(outofmtotalfeatures)thatarerelatedwithaspecificclass.They
considerexplainabilityofaclassifierasaproblemoffaultlocationincombinatorial
testing.Specifically,theyseektoidentifycombinationsoffeaturesthatarepresentin
membersofagivenclassbutabsentorrareinnon-members.However,findingagroup
ofnfeaturesoutofmexistingfeaturesinvolvesm-choose-ncombinations.Thisdoes
123

Sparseobliquedecisiontrees:atooltounderstand… 2891
notscaletoneuralnetworkssuchasLeNet5orVGG16,whichusealargenumberof
features.
Ourresultsapplytofeaturesextractedbyadeepnetwithspecificweights.Since
deepnetsaretypicallyoverparameterizedandhavelocaloptima,itispossibletoobtain
numerically very different weights depending on the initialization and optimization
protocol. We have not explored how this may affect the resulting features, tree and
masks. We did observe that our results seem robust to the initialization of the tree
itself.
Onefaircriticismofmimic-basedinterpretationsisthat,inlearningamimic,what
we are interpreting is the mimic, not the original model (here, the neural net). For
example,ifthemimicstronglyusesfeaturei toclassifyintoclassk (say),thenitis
temptingtoconcludethattheoriginalmodelalsodoesthat.Thisisrisky,especially
if the mimic is not very accurate with respect to the neural net. However, in our
experimentswetransferredtheinsightsobtainedthroughthetreemimictotheoriginal
neuralnet(e.g.inmaskingfeatures)andconfirmedthattheystillholdthereformost
ofthetrainingandtestinstances.
8 Conclusion
Ourpaperdemonstratestheuseofsparseobliquedecisiontreesasapowerful“micro-
scope”toinvestigatethebehaviorofdeepnets,bylearninginterpretableyetaccurate
treesthatmimictheclassifierpartofadeepnet.Thetreetakesasinputtheneuralnet
“features” produced by the neural net for a given input instance (that is, the neural
netactivationsataninternallayer).Thetreethenpredictsthecorrespondingclassfor
thosefeatures,emulatingtheclassificationbehavioroftheneuralnetwithveryhigh
accuracy, at least in the neural nets we considered. Using oblique trees trained with
theTAOalgorithmiscriticalforthistosucceed.
The resulting tree gives insights about the relation between neurons and classes,
suchaswhatgroupsofneuronsareinvolvedinpredictingwhatclasses.Italsoenables
thedesignofsimplemanipulationsoftheneuronactivationsthatcan,foranytraining
ortestinstance,changetheclasspredictedinvarious,controllableways(thusmaking
adversarialattackspossibleatthelevelofthedeepnetfeatures).
This approach to interpreting or manipulating features applies to other types of
deepnetsanddata,suchasaudioorlanguage.Itmayalsoprovehelpfulinotherareas
wheredeepnetsarebeingapplied,suchasinbiology,wherewemaybeabletorelate
neurons to genes or diseases, and observe the effect of “knocking out” such genes,
whichisessentiallywhatourproposedmasksdo.
Acknowledgements WorkpartiallysupportedbyNSFawardIIS–2007147.
AppendixA:Neuralnetworkarchitectures
Tables 2 and 3 describe the architectures for our VGG16 network and the LeNet5
network.
123

| 2892 |     |     |     |     |     |     |     | S.S.Hadaetal. |
| ---- | --- | --- | --- | --- | --- | --- | --- | ------------- |
)2=edirts(wodniw2
|     |                    | noitazilamroNhctaBybdewollof |                        | noitazilamroNhctaBybdewollof | noitazilamroNhctaBybdewollof |     |                    | noitazilamroNhctaBybdewollof |
| --- | ------------------ | ---------------------------- | ---------------------- | ---------------------------- | ---------------------------- | --- | ------------------ | ---------------------------- |
|     | sretlfi3           |                              | sretlfi3               | sretlfi3                     |                              |     | sretlfi3           |                              |
|     |                    | × )1=gniddap,1=edirts(       | × )1=gniddap,1=edirts( | ×                            | )1=gniddap,1=edirts(         |     | ×                  | )1=gniddap,1=edirts(         |
|     | 3215,lanoitulovnoc |                              | 3215,lanoitulovnoc     | 3215,lanoitulovnoc           |                              |     | 3215,lanoitulovnoc |                              |
|     | ytivitcennoC       |                              |                        |                              |                              | ×   |                    |                              |
2,loopxam
|     |     | ULeR |     | ULeR |     | ULeR |     | ULeR |
| --- | --- | ---- | --- | ---- | --- | ---- | --- | ---- |
|     |     | →    |     | →    |     | →    |     | →    |
reyaL
| )ezisegami46 | 11  |     | 21  | 31  |     | 41  | 51  |     |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
×
46(tesbusteNegamIruoroftenlaruen61GGVdefiidomruofoerutcetihcrA
)2=edirts(wodniw2
|     |                   | noitazilamroNhctaBybdewollof |                      | noitazilamroNhctaBybdewollof |     |                        | noitazilamroNhctaBybdewollof | noitazilamroNhctaBybdewollof |
| --- | ----------------- | ---------------------------- | -------------------- | ---------------------------- | --- | ---------------------- | ---------------------------- | ---------------------------- |
|     |                   |                              |                      |                              |     | sretlfi3               | sretlfi3                     |                              |
|     | sretlfi3          |                              | sretlfi3             |                              |     |                        |                              |                              |
|     |                   | )1=gniddap,1=edirts(         | )1=gniddap,1=edirts( |                              |     | × )1=gniddap,1=edirts( | ×                            | )1=gniddap,1=edirts(         |
|     |                   | ×                            | ×                    |                              |     | 3821,lanoitulovnoc     | 3821,lanoitulovnoc           |                              |
|     | 346,lanoitulovnoc |                              | 346,lanoitulovnoc    |                              |     |                        |                              |                              |
egamI3
×
|     | ytivitcennoC |      |     | 2,loopxam |     |     |      |      |
| --- | ------------ | ---- | --- | --------- | --- | --- | ---- | ---- |
|     |              | ULeR |     | ULeR      |     |     | ULeR | ULeR |
×
46
×
|     | 46  | →   |     | →   |     |     | →   | →   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
2elbaT
reyaL tupnI
|     | 1   |     | 2   | 3   |     | 4   | 5   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
123

Sparseobliquedecisiontrees:atooltounderstand… 2893
)6.0=p(tuoporD )6.0=p(tuoporD
| noitazilamroNhctaBybdewollof | noitazilamroNhctaBybdewollof              |                    |
| ---------------------------- | ----------------------------------------- | ------------------ |
| sretlfi3                     | sretlfi3                                  |                    |
|                              | noruen6904reyaLesneD noruen6904reyaLesneD |                    |
| × )1=gniddap,1=edirts(       | × )1=gniddap,1=edirts(                    | noruen61reyaLesneD |
→ →
| 3215,lanoitulovnoc | 3215,lanoitulovnoc |     |
| ------------------ | ------------------ | --- |
ULeRybdewollof ULeRybdewollof
ytivitcennoC
| ULeR | ULeR |     |
| ---- | ---- | --- |
xamtfos
→ →
reyaL
| 61                     | 71 81 91                                                  | 02 12                                          |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| )2=edirts(wodniw2      | noitazilamroNhctaBybdewollof noitazilamroNhctaBybdewollof | noitazilamroNhctaBybdewollof )2=edirts(wodniw2 |
|                        | sretlfi3 sretlfi3 sretlfi3                                |                                                |
|                        | × )1=gniddap,1=edirts( × )1=gniddap,1=edirts(             | × )1=gniddap,1=edirts(                         |
|                        | 3652,lanoitulovnoc 3652,lanoitulovnoc 3652,lanoitulovnoc  |                                                |
| ×                      |                                                           | ×                                              |
| ytivitcennoC 2,loopxam |                                                           | 2,loopxam                                      |
|                        | ULeR ULeR                                                 | ULeR                                           |
|                        | → →                                                       | →                                              |
deunitnoc
2elbaT
reyaL
01
| 6   | 7 8 9 |     |
| --- | ----- | --- |
123

2894 S.S.Hadaetal.
Table3 LeNet5architecture
BlocknameBlockdescription
conv1 convolutionwithkernelsize5 5and20channelsReLUmaxpoolingwithkernelsize2 2
× ×
conv2 convolutionwithkernelsize5 5and50channelsReLUmaxpoolingwithkernelsize2 2
× ×
fc1 Fullyconnectedlayerwith500hiddenunitsReLU
dropout p=0.5
fc2 Fullyconnectedlayerwith10hiddenunits
AppendixB:Controlexperiments
CARTtreesareunsuitableasmimicsforthedeepnet
SparseobliquetreestrainedwithTAOhavebeenshown(Carreira-PerpiñánandTaval-
lali 2018) to outperform traditional tree learning algorithms by a large margin, in
particularaxis-alignedtreestrainedwithCART(Breimanetal.1984;Therneauetal.
2019; Pedregosa et al. 2011). Still, we tried to construct a mimic by using an axis-
aligned tree trained with CART. In an axis-aligned tree, each decision node tests a
single feature (rather than a linear combination) in order to send an instance down
its left or right child. We used the CART implementation of scikit-learn (Pedregosa
etal.2011).AsiscustomarywithCART,wefirstallowthetreetogrowinfulland
then apply cost-complexity pruning, choosing the best pruning hyperparameter by
cross-validation.WelearnedtheCARTtreeonthesamedatasetofVGG16features
astheTAOtree.Table4showsstatisticsoftheresultingtreeandFig.14thetreeitself.
Itisobviousthatthetreeisbothgrosslyinaccurateintesterror(21.2%)andhugein
size(depth54and1381nodes).ThismakesitunsuitableasamimicforVGG16and
practicallyimpossibletointerpretortoconstructmasks.
Rotateddeepnetfeaturesdonotadmitsparsitywell
Our finding that a very small subset of deep net features are sufficient to control
classificationforagivenclassissurprising,becauseitisnotstrictlynecessaryforthis
to happen. That is, each class could be dependent on the collaborative values of all
(ornearlyall)features.Toverifythis,wemanufacturedaversionofthefeaturesthat
contains all the information in the original features, but mixes them in a dense way
asalinearcombination.Specifically,wemultipliedtheoriginalfeaturesf byadense,
invertible matrix Q, to obtain transformed features f Qf. Clearly, both the fully-
=
connectedlayersofVGG16oranobliquetreecanabsorbthistransformation,namely
byusinganewmatrixW WQ 1inthefirstfully-connectedlayerorbyusinganew
−
weight vector w Q
T=
w in each decision node i of the tree, respectively (since
i − i
thenwehaveWf = WfandwTf wTf,whereWandw weretheoriginalmatrixor
= i = i i
weightvector).However,thisneednotbetrueanymoreifweforcetheweightvector
w tobesparsebyusingalargeenoughλvalue.(Thesamewouldbetrueforthedeep
i
netifpruningweightsinthefully-connectedlayers.)Hence,weexpectedthattraining
123

Sparseobliquedecisiontrees:atooltounderstand… 2895
OAThtiweerteuqilboesraps,TRAChtiweertdengila-sixa:serutaef61GGVnoeertagniniarT
4elbaT
OAT
TRAC
33
λ
1
λ
10.0
λ
gninurperofeB
gninurpretfa
=
=
=
97.1
00.0
00.0
00.0
79.1
)%(rorregniniarT
65.9
19.7
36.7
74.12
42.12
)%(rorretseT
804
6631
3244
597
916
)2918fotuo(desuserutaeF
5
6
6
75
45
htpeD
13
93
15
5871
1831
sedonforebmuN
123

2896 S.S.Hadaetal.
Fig.14 CARTaxis-alignedtreetrainedonVGG16features
a sparse oblique tree on the rotated VGG16 features would fail to produce a tree as
sparse as before but with low test error. Indeed this is what happened, as described
next.
TogeneratearotatedversionoftheVGG16features,weusedadenseorthogonal
matrix as Q matrix 5. This has the desired effect of mixing all the features in an
invertible way. Without sparsity (λ 0), the TAO tree achieves identical error to
=
the unrotated features’ tree (as predicted theoretically). As we increase λ in order
to achieve sparsity,the behavior of the treeis very different to that of Fig.4,where
thetreesize,numberofnonzerosandtraining/testerrorchangecontinuouslywithλ.
Instead,increasingthevalueofλoverawiderange(includingthevaluesusedforour
originaltrees)resultsinnosparsityatall.Butoncewereachλ 45,thetreechanges
=
drastically:itbecomesquitesparsebutitstesterrorjumpsto11.3%,muchhigherthan
withtheunrotatedfeatures.
ThisshowsthatthefeatureslearntbyVGGarespecialinthattheyseemtooperate
insmallgroupsassociatedwithclasses,ratherthanallormostfeaturesparticipating
ineachclass.SinceVGG16couldlearnmixedratherthansparsefeatures,thereason
mustbenotinthearchitectureofVGG16butinthetrainingalgorithmand/orobjective
function.
5 This satisfies Q 1 QT and is easier to handle. We use the default matrix in Matlab’s
− =
gallery(’orthog’,n,k) function, which is an n n matrix with entries defined as qij
× =
2 sin πij .
n 1 n 1
+ +
√ ( )
123

Sparseobliquedecisiontrees:atooltounderstand… 2897
Table5 TAOsparseobliquetreestrainedwiththeoriginalVGGfeaturesandtherotatedVGGfeatures
Originalfeatures
λ #featuresselectedbythetreeoutof8192 Trainingerror(%) Testerror(%)
| 0.01 4423 |     | 0.00 | 7.63 |
| --------- | --- | ---- | ---- |
| 1 1366    |     | 0.00 | 7.91 |
| 33 408    |     | 1.79 | 9.56 |
Rotatedfeatures
λ #featuresselectedbythetreeoutof8192 Trainingerror(%) Testerror(%)
| 1 8192  |         | 0.00    | 7.91    |
| ------- | ------- | ------- | ------- |
| 45 335  |         | 1.91    | 11.34   |
| k=0 k=1 | k=2 k=3 | k=4 k=5 | k=6 k=7 |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC | ABC ABC | ABC ABC | ABC ABC |
| ------- | ------- | ------- | ------- |
| D D     | D D     | D D     | D D     |
| E F E F | EF E F  | E F E F | E F E F |
0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF
Modified net Modified net Modified net Modified net Modified net Modified net Modified net Modified net
| k=8 k=9 | k=A k=B | k=C k=D | k=E k=F |
| ------- | ------- | ------- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC   | ABC ABC  | ABC ABC  | ABC ABC  |
| --------- | -------- | -------- | -------- |
| D EF D EF | D EF D E | D E D EF | D EF D E |
0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF F 0123456789ABCDEF F 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF F 0123456789ABCDEF
Modified net Modified net Modified net Modified net Modified net Modified net Modified net Modified net
Fig.15 ConfusionmatricesforVGG(testset)usingtheAll to classkmaskcreatedwithoursparse
obliquetrees
LinearclassifierandCARTtreesarenotsuitableforexplanation
Insteadofsparseobliquetrees,onecoulduseotherinterpretablemodelstotrytounder-
standtherelationshipbetweentheclassesandtheneurons.Weshowresults(onthe
VGG16features)usingtwoothermodelsthatarewidelyconsideredasinterpretable:
alinearclassifierandanaxis-alignedtree(trainedusingCART).Asshownnext,they
donotworknearlyaswell.
Softmax linear classifier First, we train a model without  regularization (so the
1
weightsarenotsparse).Thisgivesareasonablygoodmimic,withatrainingerrorof
0.52%andatesterrorof8.04%.However,allfeaturesparticipateinallclasses,which
makesthemodeldifficulttointerpret.Next,wetrainamodelwith 1 regularization
andtunethelattertoachieveasimilarsparsity(around83%)asourtree(Fig.5).The
linearclassifierachievesatrainingerrorof1.12%andatesterrorof9.84%,which
is worse but not too far from our tree (0% train and 7.63% test error). However,
wecannotfindclass-specificneuronsinthislinearclassifier.Thisisevidentfrom
Fig.16,whereweshowtheresultsoftheAll to classk maskcreatedbyclass-
specific neurons from the linear model. We can see that the mask fails for every
class,meaningthelinearmodelcannotidentifytheclass-specificneurons.
123

| 2898 |         |     |         | S.S.Hadaetal. |
| ---- | ------- | --- | ------- | ------------- |
| k=0  | k=1 k=2 | k=3 | k=4 k=5 | k=6 k=7       |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC | ABC | ABC | ABC ABC | ABC ABC |
| ------- | --- | --- | ------- | ------- |
| D D     | D   | D   | D D     | D D     |
| EF E F  | E F | EF  | EF E F  | E F E F |
012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF
| k=8 | k=9 k=A | k=B | k=C k=D | k=E k=F |
| --- | ------- | --- | ------- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC | ABC | ABC  | ABC ABC | ABC ABC |
| ------- | --- | ---- | ------- | ------- |
| D E D E | D E | D EF | D E D E | D E D E |
F 0123456789ABCDEF F 012345Mo6dif7ied8 n9etABCDEF F 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF F 0123456789ABCDEF F 0123456789ABCDEF F 012345Mo6dif7ied8 n9etABCDEF F 012345Mo6dif7ied8 n9etABCDEF
| Modified net |     |     | Modified net Modified net |     |
| ------------ | --- | --- | ------------------------- | --- |
Fig.16 LikeFig.15butusingasparselinearclassifier
| k=0 | k=1 k=2 | k=3 | k=4 k=5 | k=6 k=7 |
| --- | ------- | --- | ------- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC | ABC | ABC | ABC ABC   | ABC ABC  |
| ------- | --- | --- | --------- | -------- |
| D E D E | D E | D E | D EF D EF | D E D EF |
| F F     | F   | F   |           | F        |
012345Mo6dif7ied8 n9etABCDEF 0123456789ABCDEF Modified net 0123456789ABCDEF Modified net 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 012345Mo6dif7ied8 n9etABCDEF 0123456789ABCDEF Modified net
| k=8 | k=9 k=A | k=B | k=C k=D | k=E k=F |
| --- | ------- | --- | ------- | ------- |
0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789 0123456789
ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO ten lanigirO
| ABC ABC     | ABC   | ABC   | ABC ABC    | ABC ABC     |
| ----------- | ----- | ----- | ---------- | ----------- |
| D E F D E F | D E F | D E F | D E F D EF | D E F D E F |
0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF 012345Mo6dif7ied8 n9etABCDEF 0123456789ABCDEF 012345Mo6dif7ied8 n9etABCDEF
Modified net Modified net Modified net Modified net Modified net Modified net
Fig.17 LikeFig.15butusinganaxis-alignedtree
CARTaxis-alignedtreeWeusethetreefromFig.14,whichhasatrainingerrorof
1.97%andatesterrorof21.34%.Obviously,thisisabadmimicofthenetwork’s
classifier,and,asshowninFig.14,itisimpossibletointerpretmanually.AsFig.17
| shows,theAll | to classk | maskalsofails. |     |     |
| ------------ | --------- | -------------- | --- | --- |
References
AdebayoJ,GilmerJ,MuellyM,GoodfellowI,HardtM,KimB(2018)Sanitychecksforsaliencymaps.
In:BengioS,WallachH,LarochelleH,GraumanK,Cesa-BianchiN,GarnettR(eds)Advancesin
NeuralInformationProcessingSystems(NEURIPS).MITPress,Cambridge
AndrewsR,DiederichJ,TickleAB(1995)Surveyandcritiqueoftechniquesforextractingrulesfrom
trainedartificialneuralnetworks.Knowl-BasedSyst8(6):373–389
BachS,BinderA,MontavonG,KlauschenF,MüllerKR,SamekW(2015)Onpixel-wiseexplanationsfor
non-linearclassifierdecisionsbylayer-wiserelevancepropagation.PLoSONE10(7):e0130140
BaesensB,SetionoR,MuesC,VanthienenJ(2003)Usingneuralnetworkruleextractionanddecision
tablesforcredit-riskevaluation.ManageSci49(3):255–350
BauD,ZhouB,KhoslaA,OlivaA,TorralbaA(2017)Networkdissection:Quantifyinginterpretability
ofdeepvisualrepresentations.In:Proceedingsofthe2017IEEEComputerSocietyConf.Computer
VisionandPatternRecognition(CVPR’17),Honolulu,HI,pp6541–6549
BreimanL(2001)Randomforests.MachLearn45(1):5–32
BreimanLJ,FriedmanJH,OlshenRA,StoneCJ(1984)Classificationandregressiontrees.Wadsworth,
Belmont
Carreira-PerpiñánMÁ(2022)TheTreeAlternatingOptimization(TAO)algorithm:Anewwaytolearn
decisiontreesandtree-basedmodels,arXiv
123

Sparseobliquedecisiontrees:atooltounderstand… 2899
Carreira-Perpiñán MÁ, Hada SS (2021) Counterfactual explanations for oblique decision trees: exact,
efficientalgorithms.In:Proc.ofthe35thAAAIConferenceonArtificialIntelligence(AAAI2021),
Online,pp6903–6911
Carreira-PerpiñánMÁ,IdelbayevY(2018)“Learning-compression”algorithmsforneuralnetpruning.In:
Proc.ofthe2018IEEEComputerSocietyConf.ComputerVisionandPatternRecognition(CVPR’18),
SaltLakeCity,UT,pp8532–8541
Carreira-PerpiñánMÁ,TavallaliP(2018)Alternatingoptimizationofdecisiontrees,withapplicationto
learningsparseobliquetrees.In:BengioS,WallachH,LarochelleH,GraumanK,Cesa-BianchiN,
GarnettR(eds)Advancesinneuralinformationprocessingsystems(NEURIPS).MITPress,Cam-
bridge
Carreira-PerpiñánMÁ,ZharmagambetovA(2020)EnsemblesofbaggedTAOtreesconsistentlyimprove
overrandomforests,AdaBoostandgradientboosting.In:Proc.ofthe2020ACM-IMSFoundations
ofDataScienceConference(FODS2020),Seattle,WA,pp35–46
Carrillo-Reid L, Han S, Yang W, Akrouh A, Yuste R (2019) Controlling visually guided behavior by
holographicrecallingofcorticalensembles.Cell178(2):447-457.e5
ChenT,GuestrinC(2016)XGBoost:ascalabletreeboostingsystem.In:Proc.ofthe22ndACMSIGKDD
Int.Conf.KnowledgeDiscoveryandDataMining(SIGKDD2016),SanFrancisco,CA,pp785–794
CravenM,ShavlikJW(1994)Usingsamplingandqueriestoextractrulesfromtrainedneuralnetworks.
In:Proc.ofthe11thInt.Conf.MachineLearning(ICML’94),pp37–45
CravenM,ShavlikJW(1996)Extractingtree-structuredrepresentationsoftrainednetworks.In:Touretzky
DS,MozerMC,HasselmoME(eds)Advancesinneuralinformationprocessingsystems(NIPS).MIT
Press,Cambridge
DattaA,SenS,ZickY(2016)Algorithmictransparencyviaquantitativeinputinfluence:theoryandexper-
imentswithlearningsystems.In:IEEESymposiumonSecurityandPrivacy(SP2016),pp598–617
DengJ,DongW,SocherR,LiLJ,LiK,Fei-FeiL(2009)ImageNet:alarge-scalehierarchicalimage
database.In:Proc.ofthe2009IEEEComputerSocietyConf.ComputerVisionandPatternRecognition
(CVPR’09),Miami,FL,pp248–255
DomingosP(1998)Knowledgediscoveryviamultiplemodels.IntellDataAnal2(1–4):187–202
DosovitskiyA,BroxT(2016)Invertingvisualrepresentationswithconvolutionalnetworks.In:Procof
the2016IEEEComputerSocietyConf.ComputerVisionandPatternRecognition(CVPR’16),Las
Vegas,NV
FanRE,ChangKW,HsiehCJ,WangXR,LinCJ(2008)LIBLINEAR:alibraryforlargelinearclassification.
JMachLearnRes9:1871–1874
FinlaysonSG,BowersJD,ItoJ,ZittrainJL,BeamAL,KohaneIS(2019)Adversarialattacksonmedical
machinelearning.Science363(6433):1287–1289
FongRC,VedaldiA(2017)Interpretableexplanationsofblackboxesbymeaningfulperturbation.In:Proc
16thIntConfComputerVision(ICCV’17),Venice,Italy,pp3449–3457
FuL(1994)Rulegenerationfromneuralnetworks.IEEETransSystManCybern24(8):1114–1124
GhorbaniA,AbidA,ZouJ(2019)Interpretationofneuralnetworkisfragile.In:Procofthe33rdAAAI
ConferenceonArtificialIntelligence(AAAI2019),Honolulu,HI,pp3681–3688
GuidottiR,MonrealeA,RuggieriS,TuriniF,GiannottiF,PedreschiD(2018)Asurveyofmethodsfor
explainingblackboxmodels.ACMComputSurv51(5):93
HadaSS,Carreira-PerpiñánMÁ(2019)Samplingthe“inverseset”ofaneuron:anapproachtounderstanding
neuralnets,arXiv:1910.04857
HadaSS,Carreira-PerpiñánMÁ(2021)Exploringcounterfactualexplanationsforclassificationandregres-
siontrees.In:ECMLPKDD3rdInt.WorkshopandTutorialoneXplainableKnowledgeDiscoveryin
DataMining(XKDD2021),pp489–504
HadaSS,Carreira-PerpiñánMÁ,ZharmagambetovA(2021)Understandingandmanipulatingneuralnet
featuresusingsparseobliqueclassificationtrees.In:IEEEIntConfImageProcessing(ICIP2021),
Online,pp3707–3711
HastieT,TibshiraniR,WainwrightM(2015)StatisticalLearningwithsparsity:theLassoandgeneraliza-
tions.Monographsonstatisticsandappliedprobability.Chapman&Hall/CRC,London
HeK,ZhangX,RenS,SunJ(2016)Deepresiduallearningforimagerecognition.In:Procofthe2016
IEEEComputerSocietyConfComputerVisionandPatternRecognition(CVPR’16),LasVegas,NV,
pp770–778
123

2900 S.S.Hadaetal.
JensenCA,ReedRD,MarksRJII,El-SharkawiMA,JungJB,MiyamotoRT,AndersonGM,EggenCJ
(1999)Inversionoffeedforwardneuralnetworks:algorithmsandapplications.ProcIEEE87(9):1536–
1549
Kindermann J, Linden A (1990) Inversion of neural networks by gradient descent. Parallel Comput
14(3):277–286
KohPW,LiangP(2017)Understandingblack-boxpredictionsviainfluencefunctions.In:Procofthe34th
IntConfMachineLearning(ICML2017,(ed)PrecupD,TehYW.Australia,Sydney,pp1885–1894
KuhnR,KackerR(2019)Anapplicationofcombinatorialmethodsforexplainabilityinartificialintelligence
andmachinelearning,draftwhitepaper,NationalInstituteofStandardsandTechnology
KumarIE,VenkatasubramanianS,ScheideggerC,FriedlerS(2020)ProblemswithShapley-value-based
explanationsasfeatureimportancemeasures.In:DauméIIIH,SinghA(eds)Procofthe37thInt.
Conf.MachineLearning(ICML2020),Online,pp5491–5500
LeCunY,BottouL,BengioY,HaffnerP(1998)Gradient-basedlearningappliedtodocumentrecognition.
ProcIEEE86(11):2278–2324
LundbergSM,LeeSI(2017)Aunifiedapproachtointerpretingmodelpredictions.In:GuyonI,Luxburg
VU,BengioS,WallachH,FergusR,VishwanathanS,GarnettR(eds)Advancesinneuralinformation
processingsystems(NIPS).MITPress,Cambridge
MahendranA,VedaldiA(2016)Visualizingdeepconvolutionalneuralnetworksusingnaturalpre-images.
IntJComputVis120(3):233–255
MarshelJH,KimYS,MachadoTA,QuirinS,BensonB,KadmonJ,RajaC,ChibukhchyanA,Ramakrishnan
C,InoueM,ShaneJC,McKnightDJ,YoshizawaS,KatoHE,GanguliS,DeisserothK(2019)Cortical
layer-specificcriticaldynamicstriggeringperception.Science365(6453):eaaw5202
McCormickK,AbbottD,BrownMS,KhabazaT,MutchlerSR(2013)IBMSPSSmodelercookbook.Packt
Publishing,Birmingham
MerrickL,TalyA(2020)Theexplanationgame:ExplainingmachinelearningmodelsusingShapleyvalues.
In:IntCross-DomainConfforMachineLearningandKnowledgeExtraction(CD-MAKE2020),pp
17–38
MontavonG,LapuschkinS,BinderA,SamekW,MüllerKR(2016)Explainingnonlinearclassification
decisionswithdeepTaylordecomposition.PatternRecogn65:211–222
MontavonG,SamekW,MüllerKR(2018)Methodsforinterpretingandunderstandingdeepneuralnetworks.
DigitalSignalProcess73:1–15
MuJ,AndreasJ(2020)Compositionalexplanationsofneurons.In:LarochelleH,RanzatoM,HadsellR,
BalcanMF,LinH(eds)AdvancesinNeuralinformationprocessingsystems(NEURIPS).MITPress,
Cambridge
MurthySK,KasifS,SalzbergS(1994)Asystemforinductionofobliquedecisiontrees.JArtifIntellRes
2:1–32
NguyenA,DosovitskiyA,YosinskiJ,BroxT,CluneJ(2016)Synthesizingthepreferredinputsforneurons
inneuralnetworksviadeepgeneratornetworks.In:LeeDD,SugiyamaM,vonLuxburgU,GuyonI,
GarnettR(eds)Advancesinneuralinformationprocessingsystems(NIPS).MITPress,Cambridge
NguyenA,CluneJ,BengioY,DosovitskiyA,YosinskiJ(2017)Plug&playgenerativenetworks:conditional
iterativegenerationofimagesinlatentspace.In:Procofthe2017IEEEComputerSocietyConf
ComputerVisionandPatternRecognition(CVPR’17),Honolulu,HI,pp3510–3520
PedregosaF,VaroquauxG,GramfortA,MichelV,ThirionB,GriselO,BlondelM,PrettenhoferP,WeissR,
DubourgV,VanderplasJ,PassosA,CournapeauD,BrucherM,PerrotM,DuchesnayÉ(2011)Scikit-
learn:MachinelearninginPython.JMachineLearningResearch12:2825–2830,availableonlineat
https://scikit-learn.org
PruthiG,LiuF,SundararajanM,KaleS(2020)Estimatingtrainingdatainfluencebytracinggradient
descent.In:LarochelleH,RanzatoM,HadsellR,BalcanMF,LinH(eds)Advancesinneuralinfor-
mationprocessingsystems(NEURIPS).MITPress,Cambridge
Qi Z, Khorram S, Fuxin L (2020) Visualizing deep networks by optimizing with integrated gradients.
In:Procofthe34thAAAIConferenceonArtificialIntelligence(AAAI2020),NewYork,NY,pp
11890–11898
QuinlanJR(1993)C4.5:programsformachinelearning.MorganKaufmann
RahwanI,CebrianM,ObradovichN(2019)Machinebehaviour.Nature568(7753):477–486
RibeiroMT,SinghS,GuestrinC(2016)“WhyshouldItrustyou?”:Explainingthepredictionsofany
classifier. In: Proc of the 22nd ACM SIGKDD Int Conf Knowledge Discovery and Data Mining
(SIGKDD2016),SanFrancisco,CA,pp1135–1144
123

Sparseobliquedecisiontrees:atooltounderstand… 2901
RibeiroMT,SinghS,GuestrinC(2018)Anchors:High-precisionmodel-agnosticexplanations.In:Proc.of
the32ndAAAIConferenceonArtificialIntelligence(AAAI2018),NewOrleans,LA,pp1527–1535
RudinC(2019)Stopexplainingblackboxmachinelearningmodelsforhighstakesdecisionsanduse
interpretablemodelsinstead.NatMachIntell1(5):206–215
SelvarajuRR,CogswellM,DasA,VedantamR,ParikhD,BatraD(2017)Grad-CAM:visualexplana-
tionsfromdeepnetworksviagradient-basedlocalization.In:Proc.16thInt.Conf.ComputerVision
(ICCV’17),Venice,Italy,pp618–626
ShrikumarA,GreensideP,KundajeA(2017)Learningimportantfeaturesthroughpropagatingactivation
differences.In:Procofthe34thIntConfMachineLearning(ICML2017),(ed)PrecupD,TehYW.
Australia,Sydney,pp3145–3153
SimonyanK,ZissermanA(2015)Verydeepconvolutionalnetworksforlarge-scaleimagerecognition.In:
Procofthe3rdIntConfLearningRepresentations(ICLR2015),SanDiego,CA
SimonyanK,VedaldiA,ZissermanA(2014)Deepinsideconvolutionalnetworks:visualisingimageclas-
sificationmodelsandsaliencymaps.In:Procofthe2ndIntConfLearningRepresentations(ICLR
2014),Banff,Canada
SinghC,MurdochWJ,YuB(2019)Hierarchicalinterpretationsforneuralnetworkpredictions.In:Procof
the7thIntConfLearningRepresentations(ICLR2019),NewOrleans,LA
ŠtrumbeljE,KononenkoI(2014)Explainingpredictionmodelsandindividualpredictionswithfeature
contributions.KnowlInfSyst41(3):647–665
SundararajanM,TalyA,YanQ(2017)Axiomaticattributionfordeepnetworks,arXiv:1703.01365
TherneauT,AtkinsonB,RipleyB(2019)rpart:recursivepartitioningandregressiontrees.Rpackage
version4.1-15,availableonlineathttps://cran.r-project.org/package=rpart
TowellGG,ShavlikJW(1993)Extractingrefinedrulesfromknowledge-basedneuralnetworks.Mach
Learn13(1):71–101
VedaldiA,SoattoS(2008)Quickshiftandkernelmethodsformodeseeking.In:Proc10thEuropeanConf
ComputerVision(ECCV’08),(ed)ForsythD,TorrP,ZissermanA.Marseille,France,pp705–718
Wei D, Zhou B, Torralba A, Freeman W (2015) Understanding intra-class knowledge inside CNN,
arXiv:1507.02379
YehCK,KimJS,YenIEH,RavikumarP(2018)Representerpointselectionforexplainingdeepneural
networks. In: Bengio S, Wallach H, Larochelle H, Grauman K, Cesa-Bianchi N, Garnett R (eds)
Advancesinneuralinformationprocessingsystems(NEURIPS).MITPress,Cambridge
ZechJR,BadgeleyMA,LiuM,CostaAB,TitanoJJ,OermannEK(2018)Variablegeneralizationperfor-
manceofadeeplearningmodeltodetectpneumoniainchestradiographs:Across-sectionalstudy.
PLoSMed15(11):e1002683
ZeilerMD,FergusR(2014)Visualizingandunderstandingconvolutionalnetworks.In:Proc13thEuropean
ConfComputerVision(ECCV’14),Zürich,Switzerland,pp818–833
ZhangQ,YangY,MaH,WuYN(2019)InterpretingCNNsviadecisiontrees.In:Procofthe2019IEEE
ComputerSocietyConfComputerVisionandPatternRecognition(CVPR’19),LongBeach,CA,pp
6261–6270
ZharmagambetovA,Carreira-PerpiñánMÁ(2020)Smaller,moreaccurateregressionforestsusingtree
alternatingoptimization.In:DauméIIIH,SinghA(eds)Procofthe37thIntConfMachineLearning
(ICML2020),Online,pp11398–11408
ZharmagambetovA,Carreira-PerpiñánMÁ(2021a)Learningatreeofneuralnets.In:ProcoftheIEEEInt
ConfAcoustics,SpeechandSig.Proc.(ICASSP’21),Toronto,Canada,pp3140–3144
ZharmagambetovA,Carreira-PerpiñánMÁ(2021b)Asimple,effectivewaytoimproveneuralnetclassi-
fication:Ensemblingunitactivationswithasparseobliquedecisiontree.In:IEEEInt.Conf.Image
Processing(ICIP2021),Online,pp369–373
ZharmagambetovA,GabidollaM,Carreira-PerpiñánMÁ(2021a)Improvedboostedregressionforests
throughnon-greedytreeoptimization.In:IntJConfNeuralNetworks(IJCNN’21),Virtualevent
ZharmagambetovA,GabidollaM,Carreira-PerpiñánMÁ(2021b)ImprovedmulticlassAdaBoostforimage
classification:Theroleoftreeoptimization.In:IEEEIntConfImageProcessing(ICIP2021),Online,
pp424–428
ZharmagambetovA,GabidollaM,Carreira-PerpiñánMÁ(2021c)Softmaxtree:Anaccurate,fastclassifier
whenthenumberofclassesislarge.In:MoensMF,HuangX,SpeciaL,YihSWt(eds)ProcConf
EmpiricalMethodsinNaturalLanguageProcessing(EMNLP2021),Online,pp10730–10745
123

2902 S.S.Hadaetal.
ZharmagambetovA,HadaSS,GabidollaM,Carreira-PerpiñánMÁ(2021d)Non-greedyalgorithmsfor
decisiontreeoptimization:anexperimentalcomparison.In:IntJConfNeuralNetworks(IJCNN’21),
Virtualevent
ZhouB,KhoslaA,LapedrizaA,OlivaA,TorralbaA(2016)Learningdeepfeaturesfordiscriminativelocal-
ization.In:Procofthe2016IEEEComputerSocietyConfComputerVisionandPatternRecognition
(CVPR’16),LasVegas,NV
Publisher’sNote SpringerNatureremainsneutralwithregardtojurisdictionalclaimsinpublishedmaps
andinstitutionalaffiliations.
SpringerNatureoritslicensor(e.g.asocietyorotherpartner)holdsexclusiverightstothisarticleunder
apublishingagreementwiththeauthor(s)orotherrightsholder(s);authorself-archivingoftheaccepted
manuscriptversionofthisarticleissolelygovernedbythetermsofsuchpublishingagreementandapplicable
law.
123