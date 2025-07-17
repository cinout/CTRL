#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="img_h"
#SBATCH --account=punim1623
#SBATCH --time=0-01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=180G

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
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --pretrained_ssl_model "./Experiments/imagenet100_htba_mocov2_ref_set_sd20/last.pth.tar" \
    --pretrained_linear_model "./Experiments/imagenet100_htba_mocov2_ref_set_sd20/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 20 \
    --num_views 64 \
    \
    --use_trigger_channel_removal \
    --ideal_case \
    --voted_channel_num 4 \
    --removed_channel_num 2 4 6 8 12 16 20 30 40 \
    \
    --minority_lower_bound 0.000 \
    --minority_upper_bound 0.010 \
    \
    --bd_detectors entropy \
    --in_n_detectors 1 \
    --note "seed=20, ideal case" \

    

##Log this job's resource usage stats###
my-job-stats -a -n -s