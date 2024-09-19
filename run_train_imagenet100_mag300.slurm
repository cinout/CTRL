#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="img100"
#SBATCH --account=punim1623
#SBATCH --time=0-03:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=110G

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
    --image_size 64 \
    --batch_size 128 \
    --eval_batch_size 512 \
    --linear_probe_batch_size 128 \
    \
    --epochs 800 \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --use_linear_probing \
    \
    --detect_trigger_channels \
    --channel_num 30 \
    --ignore_probe_channel_num 15 \
    \
    --pretrained_ssl_model "./Experiments/20240906_183318_46_6-imagenet100-simclr-resnet18-poi0.01-magtrain300.0-magval300.0-bs512-lr0.06-knnfreq5-SSDYes/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240906_183318_46_6-imagenet100-simclr-resnet18-poi0.01-magtrain300.0-magval300.0-bs512-lr0.06-knnfreq5-SSDYes/linear.pth.tar" \
    \
    --minority_criterion ss_score \
    --minority_percent_lower_bound 0.000 \
    --minority_percent_upper_bound 0.010 \
    \
    --use_frequency_detector \
    --frequency_detector_epochs 800 \
    \
    --ignore_probe_channels \
    --note "0.000-0.010, ep=800, channel_num=30, ignore_probe_channel_num=15" \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s