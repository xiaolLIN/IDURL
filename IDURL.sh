#!/bin/bash

gpu_id=$1
model=$2
data=$3
K=$4
disen_lambda=$5
align_lambda=$6

model_name=${model}
batch_size=256


if [ $data == "beauty" ] ; then
dataset="Amazon_Beauty"
elif [ $data == "sport" ] ; then
dataset="Amazon_Sports_and_Outdoors"
elif [ $data == "yelp" ] ; then
dataset="yelp"
elif [ $data == "toy" ] ; then
dataset="Amazon_Toys_and_Games"
else
echo "What?"
fi     #ifend


repr=1
if [ $K == "3" ] ; then
    model_config="--n_newc_repr=3"
    n_facet_all=`expr 3 + $repr`
elif [ $K == "4" ] ; then
    model_config="--n_newc_repr=4"
    n_facet_all=`expr 4 + $repr`
elif [ $K == "5" ] ; then
    model_config="--n_newc_repr=5"
    n_facet_all=`expr 5 + $repr`
fi


echo $model_config
echo $model_name

idra=1
echo ${disen_lambda}
echo ${align_lambda}
echo ${dataset}
config_files="configs/${dataset}.yaml"
if [ $data == "yelp" ] ; then
      python run_recbole.py --gpu_id=${gpu_id} --model=${model_name} --dataset=${dataset} --rm_dup_inter=None  \
              --disen_lambda=${disen_lambda} --idra=${idra} --align_lambda=${align_lambda}  \
              --train_batch_size=${batch_size} \
              --split_eval=1 --config_files=${config_files}  --n_facet_all=${n_facet_all} \
              ${model_config//+/ }
else
      python run_recbole.py --gpu_id=${gpu_id} --model=${model_name} --dataset=${dataset} \
              --disen_lambda=${disen_lambda} --idra=${idra} --align_lambda=${align_lambda}  \
              --train_batch_size=${batch_size} \
              --split_eval=1 --config_files=${config_files}  --n_facet_all=${n_facet_all} \
              ${model_config//+/ }
fi
