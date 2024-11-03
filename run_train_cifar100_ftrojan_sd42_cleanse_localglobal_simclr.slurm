#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="c100_f"
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
    --method simclr \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar100 \
    --target_class 0 \
    \
    --pretrained_ssl_model "./Experiments/20241024_103652_10_60_cifar100_ftrojan_linear_ref_set_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 1 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.99 \
    --trigger_path "20241029_170153_68_40_trigger_estimation_simclr_cifar100_ftrojan" \
    --num_clusters 100 \
    --mitigate_epochs 5 \
    --draw_local_trigger_by local \
    --note "local&global [all triggers]" \


##Log this job's resource usage stats###
my-job-stats -a -n -s