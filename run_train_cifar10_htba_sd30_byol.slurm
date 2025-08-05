#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c10_h"
#SBATCH --account=punim1623
#SBATCH --time=0-01:20:00

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
    --method byol \
    --trigger_type htba \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --linear_probe_normalize ref_set \
    \
    --seed 30 \
    --num_views 64 \
    \
    --use_trigger_channel_removal \
    --ideal_case \
    --find_channels_from_n_few_samples 8 \
    --voted_channel_num 8 \
    --removed_channel_num 40 \
    --use_ss_contribute_percent \
    \
    --minority_lower_bound 0.000 \
    --minority_upper_bound 1.000 \
    --bd_detectors entropy \
    --in_n_detectors 1 \
    \
    --note "ideal_case, use 8 images only, use SS contribution percentage" \
    --pretrained_ssl_model "./Experiments/cifar10_htba_byol_ref_set_sd30/last.pth.tar" \
    --pretrained_linear_model "./Experiments/cifar10_htba_byol_ref_set_sd30/linear.pth.tar" \

##Log this job's resource usage stats###
my-job-stats -a -n -s