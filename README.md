# landau

Use analogy with Landau theory of phase transitions to look for critical transitions in data.

Associated publication:

Identifying a developmental transition in honey bees using gene expression data  
Bryan C. Daniels, Ying Wang, Robert E. Page Jr, Gro V. Amdam  
bioRxiv 2022.11.03.514986; doi: https://doi.org/10.1101/2022.11.03.514986

The figures in the publication can be reproduced using the following notebooks:

* Figures 1, 4, 8: landau-simulation-analysis.ipynb
* Figures 2, 7: landau-bee-data-analysis.ipynb
* Figure 3, Table 1: LandauTransition-PaperDataAnalysis.nb
* Figure 5: critical-state-correlations.ipynb
* Figure 6A: hepatocellular-carcinoma-data.ipynb
* Figure 6B: LandauTransition-HCC.nb	

## Dependencies
* Python 3
* WolframScript (either through a Mathematica install or standalone)
* numpy
* scipy
* scikit-learn
* optional for running simulations: pandas, numba

## Installation

To install the package and its python dependencies using `pip`, clone the repository, descend into the `landau/` folder, and run

```
pip install -e .
```

To automatically include the optional dependencies for running test simulations, instead run

```
pip install -e '.[simulation]'
```

WolframScript must also be installed on the system.  This can be accomplished by installing Mathematica (and using the prepackaged Extras installer on Mac) or installed as a standalone package available for free download here: https://www.wolfram.com/wolframscript/ 