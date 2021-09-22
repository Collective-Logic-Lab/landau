(* ::Package:: *)

(* ::Input:: *)
(*(**)
(*LandauTransitionAnalysis.nb*)
(*Bryan Daniels*)
(**)
(*2020.11.16 Branching from LandauTransition-PaperModelAnalysis.nb*)
(**)
(*8.24.2018 Taking content from LandauTransition.nb and LandauTransition-PaperDataAnalysis.nb*)
(**)*)


(* ::Subsubsection::Initialization::Closed:: *)
(*Define distribution*)


(* ::Input::Initialization:: *)
(*see LandauTransition.nb for tests of these definitions*)


(* ::Input::Initialization:: *)
LandauTransitionDistributionRelativeLogPDF[x_,mu_,J_,nu_,c_,d_]:=-(1/2)(x-mu).J.(x-mu)-((c-1)/2)(nu.J.nu)((x-mu).nu)^2-(d/4)(nu.J.nu)^2((x-mu).nu)^4


(* ::Input::Initialization:: *)
(*for now we'll restrict nu to be along an eigenvector of J---the index of nu in Jvals is nuIndex*)
logNormalizationDiagonal[Jvals_,nuIndex_,c_,d_]:=((Length[Jvals]-1)/2)Log[2\[Pi]]+0.5(Log[Jvals[[nuIndex]]]-Plus@@Log[Jvals])+Log[If[c>0,(c Exp[(c^2/(8 d))] BesselK[1/4,c^2/(8 d)])/(Sqrt[2] Sqrt[c d Jvals[[nuIndex]]]),(Exp[(c^2/(8 d))] \[Pi] (BesselI[-(1/4),c^2/(8 d)]+BesselI[1/4,c^2/(8 d)]))/(2 Sqrt[-((d Jvals[[nuIndex]])/c)])]]


(* ::Input::Initialization:: *)
(*(Assumes any zeros come at the end of Jvals for nuIndex to work)*)LandauTransitionDistributionLogPDFdiagonal[x_,mu_,Jvals_,Jvecs_,nuIndex_,c_,d_,nuMu_]:=
With[{
J=Transpose[Jvecs].DiagonalMatrix[Jvals].Jvecs,
nu=Jvecs[[nuIndex]]},
LandauTransitionDistributionRelativeLogPDF[x, mu + nuMu nu, J,nu,c,d]-logNormalizationDiagonal[Select[Jvals,#>0&],nuIndex,c,d]
]


(* ::Input::Initialization:: *)
logNormalizationGaussian[Jvals_]:=(Length[Jvals]/2)Log[2\[Pi]]+0.5(-Plus@@Log[Jvals])


(* ::Input::Initialization:: *)
GaussianLogPDFdiagonal[x_,mu_,Jvals_,Jvecs_]:=
With[{
J=Transpose[Jvecs].DiagonalMatrix[Jvals].Jvecs,
nu=Table[1,Length[Jvals]]},
LandauTransitionDistributionRelativeLogPDF[x,mu,J,nu/Norm[nu],1,0]-logNormalizationGaussian[Select[Jvals,#>0&]]
]


(* ::Subsubsection::Initialization::Closed:: *)
(*Define Landau maximum-likelihood algorithm*)


(* ::Input::Initialization:: *)
(*copied from LandauTransition-PaperDataAnalysis.nb*)


(* ::Text::Initialization:: *)
(*output of LandauMaxLikelihood : {fitMu, fitVals, fitVecs, fitparams}*)


(* ::Input::Initialization:: *)
LandauMaxLikelihood[dataFull_,numNuMax_:10,dmin_:0.001]:=
With[
{Jinit=PseudoInverse[Covariance[dataFull](Length[dataFull]-1)/Length[dataFull]],
muInit=Mean[dataFull],
Nsamples=Length[dataFull],
Ncomponents=Length[First[dataFull]]},
With[{
x=dataFull,
(*eigenvalues of J are put in increasing order, with any zeros at the \
end*)
Jvals=PadRight[Reverse[SingularValueList[Jinit]], Ncomponents],
JvecsUnordered=Eigenvectors[Jinit]},
With[{
maxNuIndex=Count[Jvals, val_ /; val != 0],
Jvecs = Join[
  Reverse[JvecsUnordered[[;; Count[Jvals, val_ /; val != 0]]]],
  JvecsUnordered[[Count[Jvals, val_ /; val != 0] + 1 ;;]]]},
{muInit,Jvals,Jvecs,
Table[
(*Check there's nothing weird going on with imaginary numbers in vecs, at least for c=1,d=1*)
If[Re[-Sum[LandauTransitionDistributionLogPDFdiagonal[x[[i]],muInit,Jvals,Jvecs,nuIndex,1,1,0],{i,1,Nsamples}]]!=Abs[-Sum[LandauTransitionDistributionLogPDFdiagonal[x[[i]],muInit,Jvals,Jvecs,nuIndex,1,1,0],{i,1,Nsamples}]],somethingIsWrong,
(*Do the actual minimization*)
Quiet[
FindMinimum[{-Sum[LandauTransitionDistributionLogPDFdiagonal[x[[i]],muInit,Jvals,Jvecs,nuIndex,c,d,nuMu],{i,1,Nsamples}]+Sum[GaussianLogPDFdiagonal[x[[i]],muInit,Jvals,Jvecs],{i,1,Nsamples}],{d>dmin}},{{c,1},{d,1},{nuMu,0}}],
{FindMinimum::nrnum,FindMinimum::eit}]],
{nuIndex, 1, Min[maxNuIndex, numNuMax]}
]}
]
]
]


(* ::Input::Initialization:: *)
Index[item_,list_]:=First[FirstPosition[list,item]]


(* ::Subsubsection::Initialization::Closed:: *)
(*Define sampling from Landau distribution*)


(* ::Input::Initialization:: *)
LandauCDF[xMinusMu_?NumberQ,c_,d_,Jnu_]:=NIntegrate[Exp[LandauTransitionDistributionLogPDFdiagonal[y,{0},{Jnu},{{1}},1,c,d]],{y,-100/Jnu,xMinusMu}]


(* ::Input::Initialization:: *)
LandauInverseCDF[p_,c_,d_,Jnu_]:=xMinusMu/.FindRoot[LandauCDF[xMinusMu,c,d,Jnu]==p,{xMinusMu,-Jnu,Jnu}]


(* ::Input::Initialization:: *)
LandauSampleOneDimension[c_,d_,Jnu_,numSamples_]:=LandauInverseCDF[#,c,d,Jnu]&/@RandomReal[{0,1},numSamples]


(* ::Input::Initialization:: *)
GaussianSample[mu_,covVals_,covVecs_,numSamples_]:=Table[mu,{numSamples}]+(RandomVariate[NormalDistribution[],{numSamples,Length[covVals]}]Table[Sqrt[covVals],{numSamples}]).covVecs


(* ::Input::Initialization:: *)
ModifiedInverse=If[#!=0,1/#,0]&;


(* ::Input::Initialization:: *)
LandauSample[mu_,Jvals_,Jvecs_,nuIndex_,c_,d_,numSamples_]:=With[{
covVals=ReplacePart[ModifiedInverse/@Jvals,nuIndex->0],
covVecs=Jvecs,
nu=Jvecs[[nuIndex]],
Jnu=Jvals[[nuIndex]]},(*Print[c,d,Jnu,numSamples];*)
GaussianSample[mu,covVals,covVecs,numSamples]+Outer[Times,LandauSampleOneDimension[c,d,Jnu,numSamples],nu]
]
