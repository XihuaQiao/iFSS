#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
shopt -s expand_aliases

ds=voc
task=$2
step=$3
name=$4
extra=$5

path=checkpoints/step/${task}-${ds}

for ns in 1 2 5; do
  for is in 0 1 2; do
    echo "step - ${step}"
    if [ "${step}" = "1" ]; then
      inc_par="--ishot ${is} --nshot ${ns}"
        exp --method PIFS --name ${name} --test ${gen_par} ${inc_par} --step ${step} --ckpt ${path}/${name}-s${ns}-i${is}_${step}.pth ${extra}
    else
      exp --method PIFS --name ${name} --test ${gen_par} --step ${step} --ckpt ${path}/${name}_${step}.pth ${extra}
    fi
  done
done
