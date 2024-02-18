# USER
This is the implementation for paper:
> "Recommendation with Causality enhanced Natural Language Explanations." In WWW 2023. 

For the implementation of the base models, we refer to the open source [NLG4RS](https://github.com/lileipisces/NLG4RS).

## Overview
In this paper, we propose the task of debiased explainable recommendation for the first time. For solving this task, we build a principled framework to jointly correct the item- and feature-level biases, and design fault tolerant IPS mechanism and latent confounder modeling strategy to improve this framework.
<img src="https://github.com/JingsenZhang/USER/blob/master/asset/graph.png" width = "800px" align=center />

## Requirements
- Python 3.8
- Pytorch >=1.10.1

## Datasets
We use three real-world datasets, including *TripAdvisor-HongKong*, *Amazon-Movie&TV* and *Yelp Challenge 2019*. All the datasets are available at this [link](https://github.com/lileipisces/NLG4RS).

## Usage
+ **Download the codes and datasets.**
+ **Run**

For example: Run run_nete_user.py

```
python run_nete_user.py --dataset [dataset_name] --lr [learning_rate]
```

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
````
@inproceedings{zhang2023recommendation,
  title={Recommendation with Causality enhanced Natural Language Explanations},
  author={Zhang, Jingsen and Chen, Xu and Tang, Jiakai and Shao, Weiqi and Dai, Quanyu and Dong, Zhenhua and Zhang, Rui},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={876--886},
  year={2023}
}
````
If you have any questions for our paper or codes, please send an email to zhangjingsen@ruc.edu.cn.
