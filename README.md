# IDURL
The source code of our SIGIR 2025 paper [**"Towards Interest Drift-driven User Representation Learning in Sequential Recommendation"**](https://dl.acm.org/doi/10.1145/3726302.3730099)


## Preparation
We train and evaluate our IDURL using a Tesla V100 GPU with 32 GB memory. <br>
Our code requires the following python packages:

> + numpy==1.21.6
> + scipy==1.7.3 
> + torch==1.13.1+cu116
> + tensorboard==2.11.2

## Usage

We provide two datasets, i.e., Beauty `./dataset/Amazon_Beauty` and Toys `./dataset/Amazon_Toys_and_Games`.  <br>
Please download the other two datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Toys_and_Games
│   ├── Amazon_Toys_and_Games.inter
│   └── Amazon_Toys_and_Games.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user
```
Run the command`./run_IDURL.sh`. After training and evaluation, check out the results in `./run_results/`.


## Contact
If you have any questions, please send emails to Xiaolin Lin (linxiaolin2021@email.szu.edu.com).


## Credit
This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DIF-SR](https://github.com/AIM-SE/DIF-SR).

## Citation
```
@inproceedings{10.1145/3726302.3730099,
author = {Lin, Xiaolin and Pan, Weike and Ming, Zhong},
title = {Towards Interest Drift-driven User Representation Learning in Sequential Recommendation},
year = {2025},
url = {https://doi.org/10.1145/3726302.3730099},
doi = {10.1145/3726302.3730099},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1541–1551}
}
```
