#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c10_f"
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

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python -u main_train.py \
    --method mocov2 \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --pretrained_ssl_model "" \
    --pretrained_linear_model "" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 1 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.8 \
     \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --find_and_ignore_probe_channels \
    --ignore_probe_channel_num 30 \
    \
    --minority_lower_bound 0.001 \
    --minority_upper_bound 0.011 \
    \
    --bd_detectors kdist \
    --kdist_k 8 \
    --normalize_backbone_features l2 \
    --in_n_detectors 1 \
    --trigger_path trigger_estimation_mocov2_cifar10_ftrojan \
    \
    --note "[ECCV cleanse] trigger_estimation_mocov2_cifar10_ftrojan" \


##Log this job's resource usage stats###
my-job-stats -a -n -s