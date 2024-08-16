#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="mocov2"
#SBATCH --account=punim1623
#SBATCH --time=0-08:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=80G

# export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
# export MASTER_PORT=28400
# echo ">>> NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python -u main_train.py \
    --method simclr \
    --channel 1 2 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --lr 0.06 \
    --wd 0.0005 \
    --poisoning \
    --epochs 800 \
    --window_size 32 \
    --mode frequency \
    --dataset cifar10 \
    --target_class 0 \

##Log this job's resource usage stats###
my-job-stats -a -n -s