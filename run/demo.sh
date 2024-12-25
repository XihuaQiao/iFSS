#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 demo.py --num_workers 4"
shopt -s expand_aliases

ds=voc
task=$2
ckpt=$3
image=$4

gen_par="--task ${task} --dataset ${ds} --iter 1000"

echo ${ckpt}

exp --method PIFS ${gen_par} --step 1 --ckpt ${ckpt} --image ${image}