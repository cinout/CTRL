#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="cf100"
#SBATCH --account=punim1623
#SBATCH --time=0-10:00:00

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
    --mode frequency \
    --channel 1 2 \
    --window_size 32 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --poisoning \
    \
    --batch_size 512 \
    --eval_batch_size 512 \
    --linear_probe_batch_size 128 \
    \
    --epochs 800 \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar100 \
    --target_class 0 \
    \
    --use_linear_probing \
    \
    --use_mask_pruning \
    --clean_threshold 0.0 \
    \
    --detect_trigger_channels \
    --channel_num 1 2 3 4 6 8 10 12 14 16 20 24 \
    \
    --note "mask prune cifar100 [--clean_threshold 0.0]" \
    # --pretrained_ssl_model "./Experiments/20240824_124554_77_92-cifar100-simclr-resnet18-poi0.01-magtrain50.0-magval100.0-bs512-lr0.06-knnfreq5-SSDYes-numc1/last.pth.tar" \

##Log this job's resource usage stats###
my-job-stats -a -n -s