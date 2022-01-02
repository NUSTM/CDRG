# Cross-Domain Review Generation for Aspect-Based Sentiment Analysis

This repository contains code for our ACL2021 Findings paper: 

[Cross-Domain Review Generation for Aspect-Based Sentiment Analysis](https://aclanthology.org/2021.findings-acl.421.pdf)


## Datasets

The training data comes from four domains: Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D).  
For each domain transfer pairs, the unlabeled data come from a combination of training data from the two domains (ratio: 1:1).

The in-domain corpus(used for training BERT-E) come from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 

Click here to get [BERT-E](https://pan.baidu.com/s/1hNyNCyfOHzznuPbxT1LNFQ) (BERT-Extented) , and the extraction code is by0i. (Please specify the directory where BERT is stored in modelconfig.py.)

## Folder

- aspect_output: aspects and opinions extracted by double propagation.
- ds-bert：training language models for target domains and using them for generating pseudo samples.
- pseudo_output: generated pseudo samples.
- ABSA: using pseudo samples for absa task.
- raw_data: traning data and testing data for four domains.

## Usage

### 1. Using double propagation to extract aspects and opinions from target unlabeled data

### 2. Generating pseudo samples

**2.1 Training DS-BERT**

* To get target language model in ds-bert (bert_lm_models), run below code ：

```
bert-b-based:
bash ./ds-bert/run_mlm_bert_b.sh

bert-e-based:
bash ./ds-bert/run_mlm_bert_e.sh
```

**2.2 Generation**

* To generate pseudo samples in pseudo_output, run below code ：

```
bert-b-based:
bash ./ds-bert/generate_bert_b.sh

bert-e-based:
bash ./ds-bert/generate_bert_e.sh
```

### 3.Using pseudo samples for ABSA**

* To get the results for ABSA by using pseudo samples, run below code ：

```
bert-b-based:
bash ./absa/run_base_bert_b.sh

bert-e-based:
bash ./absa/run_base_bert_e.sh
```
