#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="cf10"
#SBATCH --account=punim1623
#SBATCH --time=0-20:00:00

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

### TODO: update model checkpoints
### TODO: update note
### TODO: remove mask pruning

### TODO: rerun with checkpoints first
### TODO: brings down the training time


module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python -u main_train.py \
    --method simclr \
    --trigger_type htba \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --use_linear_probing \
    \
    --pretrained_ssl_model "" \
    --pretrained_linear_model "" \
    --pretrained_frequency_model "" \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --ignore_probe_channels \
    --ignore_probe_channel_num 30 \
    \
    --use_frequency_detector \
    --frequency_detector_epochs 500 \
    \
    --minority_1st_lower_bound 0.000 \
    --minority_1st_upper_bound 0.010 \
    # \
    # --secondary_detector ss_score \
    # \
    # --minority_2nd_lower_bound 0.010 \
    # --minority_2nd_upper_bound 0.600 \
    \
    --use_mask_pruning \
    --clean_threshold 0.0 \
    \
    --note "htba" \

##Log this job's resource usage stats###
my-job-stats -a -n -s