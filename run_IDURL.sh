#!/bin/bash

# Note that we take SASRec as the default backbone model to obtain original user representations.

# for reproduction
# Amazon Beauty
#python run_recbole.py --gpu_id=0 --model=SASRec_IDURL --dataset='Amazon_Beauty' --config_files='configs/Amazon_Beauty.yaml' --split_eval=1
# Amazon_Sports_and_Outdoors
#python run_recbole.py --gpu_id=0 --model=SASRec_IDURL --dataset='Amazon_Sports_and_Outdoors' --config_files='configs/Amazon_Sports_and_Outdoors.yaml' --split_eval=1
# Amazon_Toys_and_Games
python run_recbole.py --gpu_id=0 --model=SASRec_IDURL --dataset='Amazon_Toys_and_Games' --config_files='configs/Amazon_Toys_and_Games.yaml' --split_eval=1
# yelp
#python run_recbole.py --gpu_id=0 --model=SASRec_IDURL --dataset='yelp' --rm_dup_inter=None --config_files='configs/yelp.yaml' --split_eval=1



# for hyper-parameter searching
#data=$1   # para of shell command
#for disen_lambda in 0.1 0.3 0.5 0.7 0.9;
#do
#  for align_lambda in 0.1 0.2 0.3 0.4 0.5;
#  do
#    ./IDURL.sh 0 SASRec_IDURL ${data} 4 ${disen_lambda} ${align_lambda}
#  done
#done
