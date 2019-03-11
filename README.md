# RTNs (Recurrent Transformer Networks)

> Version 1.0 (11 Mar. 2019)
>
> Contributed by Seungryong Kim (seungryong.kim@epfl.ch).

This code is written in MATLAB, and implements the RTNs [[project website](https://seungryong.github.io/RTNs/)].

## Dependencies ##
  - Download [[VLFeat](http://www.vlfeat.org/)] and [[MatConvNet](http://www.vlfeat.org/matconvnet/)].
  - Download the datasets:
    - [[Proposal Flow Benchmark](https://drive.google.com/open?id=1hEC2yxaEALouPMsUVzHp7hZ-Ka3C3sZo)];
    - [[PASCAL-2011 Part Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar)].

## Pre-trained Models ##
  - RTNs with synthetic training (in `data/RTNs_synthetic`) [[RTNs_synthetic](https://drive.google.com/open?id=14lXC1qu1HRT8X-ma5DfDqOv-7NyAgWRB)].
  - RTNs on PF-PASCAL (in `data/RTNs`) [[RTNs](https://drive.google.com/open?id=1MgINF4Q9ZAM3SLdcEOM7i8eflYjs9yM9)].

## Getting started ##
  - `get_train_pascal_synthetic.m` get the synthetic training data from PASCAL-VOC 2011 datasets.
  - `get_train_pf_pascal.m` get the training data from PF-PASCAL datasets.
  - `train_pascal_synthetic.m` trains the RTNs with the synthetic training data
  - `train_pf_pascal.m` trains the RTNs on the PF-PASCAL training data
  - `test_pf_pascal.m` tests the RTNs on the PF-PASCAL training data

## Notes ##

  - The code is provided for academic use only. Use of the code in any commercial or industrial related activities is prohibited.
  - If you use our code, please cite the paper.

```
@InProceedings{kim2018nips,
author = {Seungryong Kim and Stephen Lin and Sangryul Jeon and Dongbo Min and Kwanghoon Sohn},
title = {Recurrent Transformer Networks for Semantic Correspondence},
booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS 2018)},
year = {2018}
}
```
