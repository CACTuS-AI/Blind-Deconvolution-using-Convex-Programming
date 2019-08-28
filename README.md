# Blind Deconvolution using Convex Programming

This page provides software to generate the figures and the experiments in the paper [Blind Deconvolution using Convex Programming](https://arxiv.org/abs/1211.5608). We also provide software created by other groups which is necessary to run our own code. 

## Required Toolboxes

The following toolboxes are required to run the MATLAB scripts below. The paths to the associated directories need to be provided in the script.

- minFunc
- minFunc_2012
- Noiselet Toolbox

## Matlab scripts

We provide the Matlab scripts that generate the figures, as well as a test file that demonstrates large scale blind deconvolution using convex programming.

- Script for blind deconvolution: test.m
  - Requires all three toolboxes
  - Requires: blindDeconvolve_implicit.m

- Script to generate Figure 3 (phase transitions): generateFig3.m
  - Requires minFunc and minFunc(2012)
  - Requires: blindDeconvolve.m

- Script to generate Figure 4 (phase transitions): generateFig4.m
  - Requires minFunc and minFunc(2012)
  - Requires: blindDeconvolve.m

- Script to generate Figure 5 (recovery in the presence of noise): generateFig5a.m and generateFig5b.m
  - Requires minFunc and minFunc(2012)
  - Requires: blindDeconvolve_implicit.m

- Script to generate Figures 6, 7, and 8 (image deblurring): deblur.m
  - Requires minFunc and minFunc(2012)
  - Requires: blindDeconvolve_implicit_2D.m
  - Image data: shapes.png

## References

The Noiselet toolbox is by Professor Romberg, while the minFunc toolbox is taken from the following paper:

- M. Schmidt, "minFunc: unconstrained differentiable multivariate optimization in Matlab", 2012.

If you use either of these files in your personal work, please remember to cite these references.