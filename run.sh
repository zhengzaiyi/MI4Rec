#!/bin/sh

lambda_V=0.1
lambda_M=0.1

model=gpt2
batch_size=32
num_metas=(300)
datasets=(beauty toys sports yelp)
num_processes=8
item_logits_infer=stella
prob_norm=sigmoid
pt_lr=1e-3
ft_lr=1e-3
cold_start=0.2
skip_pt=""

for dataset in ${datasets[@]}; do
    for num_meta in ${num_metas[@]}; do
        accelerate launch --multi_gpu --num_processes=$num_processes src/warm_pt.py --dataset $dataset --model $model --lambda_V $lambda_V --batch_size $batch_size --num_meta $num_meta --item_logits_infer $item_logits_infer -p $prob_norm -lr $pt_lr -c $cold_start
        accelerate launch --multi_gpu --num_processes=$num_processes src/warm_ft.py --dataset $dataset --model $model --lambda_V $lambda_M --batch_size $batch_size --num_meta $num_meta --item_logits_infer $item_logits_infer -p $prob_norm -lr $ft_lr -c $cold_start --pt_lr $pt_lr --pt_lv $lambda_V $skip_pt
        python predict.py --dataset $dataset --model $model --lambda_V $lambda_M --num_meta $num_meta --item_logits_infer $item_logits_infer -p $prob_norm -lr $ft_lr -c $cold_start
    done
done