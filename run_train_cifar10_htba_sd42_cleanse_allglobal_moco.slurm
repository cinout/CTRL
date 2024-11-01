#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c10_h"
#SBATCH --account=punim1623
#SBATCH --time=0-01:00:00

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
    --method mocov2 \
    --trigger_type htba \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --pretrained_ssl_model "./Experiments/20241024_142323_81_27_cifar10_htba_linear_ref_set_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20241024_142323_81_27_cifar10_htba_linear_ref_set_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 1 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.99 \
    --trigger_path "20241030_090810_82_13_trigger_estimation_mocov2_cifar10_htba" \
    --num_clusters 10 \
    --mitigate_epochs 5 \
    --draw_local_trigger_by global \
    --note "closest, draw_local_by_global [mitigate 5 epochs]" \


##Log this job's resource usage stats###
my-job-stats -a -n -s