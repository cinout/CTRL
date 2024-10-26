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
#SBATCH --time=0-05:00:00

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
    --method byol \
    --trigger_type htba \
    \
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --pretrained_ssl_model "./Experiments/20241024_160737_22_62_imagenet100_htba_linear_ref_set_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20241024_160737_22_62_imagenet100_htba_linear_ref_set_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 1 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.9 \
    --use_dynamic_lam \
    --lam 0.1 \
    --trigger_path 20241026_101059_21_52_trigger_estimation_byol_imagenet100_htba \
    \
    --mitigate_epochs 1000 \
    --trigger_overlay_option 2 \
    --note "mitigate 1000 eps [trigger=2]" \

    

##Log this job's resource usage stats###
my-job-stats -a -n -s