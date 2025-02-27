#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 1"
shopt -s expand_aliases

ds=voc
task=$2

name=$3
extra=$4

path=checkpoints/step/${task}-${ds}

exp --method COS --name ${name} --epoch 0 --task ${task} --dataset ${ds} --batch_size 10 --step 0 --ckpt ${path}/COS-mixup-manifold-4_0.pth ${extra}
