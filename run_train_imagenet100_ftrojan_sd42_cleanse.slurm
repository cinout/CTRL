#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="img_f"
#SBATCH --account=punim1623
#SBATCH --time=0-18:00:00

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
    --method simclr \
    \
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --pretrained_ssl_model "./Experiments/20240906_183318_46_6_imagenet100_ftrojan_linear_regular_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240924_155056_66_73_imagenet100_ftrojan_linear_refset_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 64 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.8 \
    --trigger_path trigger_estimation_simclr_imagenet100_ftrojan \
    \
    --note "[ECCV cleanse] trigger_estimation_simclr_imagenet100_ftrojan" \

    

##Log this job's resource usage stats###
my-job-stats -a -n -s