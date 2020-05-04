# NEWMA
A new method for scalable model-free online change-point detection.

This repository contains the code for [NEWMA: a new method for scalable model-free online change-point detection](https://arxiv.org/abs/1805.08061), *Nicolas Keriven, Damien Garreau, Iacopo Poli*.

To cite this work
```
@ARTICLE{9078835, 
author={N. {Keriven} and D. {Garreau} and I. {Poli}}, 
journal={IEEE Transactions on Signal Processing}, 
title={NEWMA: a new method for scalable model-free online change-point detection}, 
year={2020}, 
volume={}, 
number={}, 
pages={1-1},} 
```

## Code
### Requirements
The code is written for Python 3.  
You can install the Python modules required by running `pip install -r requirements.txt` 
inside the folder.

### Installing the `onlinecp` package

You can install the `onlinecp` Python package by running
`pip install ./` from the root folder of this repository.

### Access to Optical Processing Units

To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/

For researchers, we also have a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information.

### Figures in the paper
You can generate data for the figures in the paper as follows:
- run `test_dim.py` and `test_B_runningtime.py` for Figure 4a
- run `test_adaptive_vs_fixed.py` for Figure 4b
- run `test_algos_synthetic_data.sh` for Figure 4c
- run `test_algos_vad.sh` for Figure 4d

The scripts to generate the plots from data are in `plots`
and they have the same name prepended by `plot_`.
Look at `plots/README.md` for info on how to run them.

### Code for old version of the paper (v1)
The code for the older version of our paper is in `code_v1`.
The subdirectory contains its `README.md`.
