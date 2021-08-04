#!/usr/bin/env wolframscript
(* ::Package:: *)

(* runLandauTransitionAnalysis.wls *)
(* Bryan Daniels *)
(* 2020/11/16 *)

(* Input: *)
(*  - Data matrix (#samples)x(#dimensions) *)
(*  - numNuMax *) 

(* Output: csv with 6 lines: *)
(*  - fitMu vector (#dimensions) *)
(*  - fitVals vector of eigenvalues (#dimensions) *)
(*  - fitVecs flattened matrix of eigenvectors (Real parts) (#dimensions x #dimensions) *)
(*  - fitVecs flattened matrix of eigenvectors (Imag parts) (#dimensions x #dimensions) *)
(*  - log-likelihoods (#nonSingularDims) *)
(*  - fit c values (#nonSingularDims) *)
(*  - fit d values (#nonSingularDims) *)
(* where #nonSingularDims = floor( #samples-1, #dimensions ) *)


<< LandauTransitionAnalysis.m


inputFilename = $ScriptCommandLine[[2]];
Print["runLandauTransitionAnalysis: Reading data from "<>inputFilename];
data = Import[inputFilename];


numNuMax = ToExpression[$ScriptCommandLine[[3]]];


Print["DEBUG: $ProcessorCount = "<>ToString[$ProcessorCount]]


Print["runLandauTransitionAnalysis: Running analysis..."];
result = LandauMaxLikelihood[data,numNuMax];


(*
Print["DEBUG: "<>ToString[Dimensions[result[[1]]]]];
Print["DEBUG: "<>ToString[Dimensions[result[[2]]]]];
Print["DEBUG: "<>ToString[Dimensions[result[[3]]]]];
Print["DEBUG: "<>ToString[Dimensions[result[[4]]]]];
*)


(*reformat output for csv*)
resultOutput = { result[[1]],
				 result[[2]],
				 Re[Flatten[result[[3]]]],
				 Im[Flatten[result[[3]]]],
				 Map[First,result[[4]]],
				 Map[c/.Last[#]&,result[[4]]],
				 Map[d/.Last[#]&,result[[4]]]
				};


outputFilename = FileBaseName[inputFilename]<>"_LTAoutput.csv";
Print["runLandauTransitionAnalysis: Writing results to "<>outputFilename];;
Export[outputFilename,resultOutput];
