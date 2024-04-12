# LogiRec


This is the implementation of our paper: [Towards High-Order Complementary Recommendation via Logical Reasoning Network](https://ieeexplore.ieee.org/abstract/document/10027703).


## Introduction

In this work, we propose a logical reasoning network: LogiRec to capture the asymmetric complementary relationship between products and seamlessly extend it to the high-order recommendation where more comprehensive and meaningful complementary relationship is learned from a query set of products. Finally, we further propose a hybrid network that is jointly optimized for learning a more generic product representation.


## Dataset

We provide the processed dataset. The data in the default folder is trained for LogiRec<sub>Hybrid</sub>  model, highOrder is for LogiRec<sub>High</sub>, and lowOrder for LogiRec<sub>Low</sub>.



## Example to run LogiRec

	bash example.sh


## Citation

	@inproceedings{wu2022towards,
  	title={Towards high-order complementary recommendation via logical reasoning network},
  		author={Wu, Longfeng and Zhou, Yao and Zhou, Dawei},
  		booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  		pages={1227--1232},
  		year={2022},
  		organization={IEEE}
	}
