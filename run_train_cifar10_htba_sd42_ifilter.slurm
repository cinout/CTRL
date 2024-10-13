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
#SBATCH --time=0-12:00:00

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
    --trigger_type htba \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --pretrained_ssl_model "./Experiments/20240920_122144_36_100_cifar10_htba_linear_regular_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240924_155056_19_48_cifar10_htba_linear_refset_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    --pretrained_frequency_model "" \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --minority_lower_bound 0.000 \
    --minority_upper_bound 0.100 \
    \
    --bd_detectors frequency_ensemble \
    --in_n_detectors 1 \
    --frequency_ensemble_size 1 \
    --frequency_detector_epochs 500 \
    --frequency_attack_trigger_ids 2 \
    \
    --seed 42 \
    \
    --num_views 64 \
    --siftout_poisoned_images \
    --note "[input_filter] single freq detector, GaussNoise with lessVariation" \

##Log this job's resource usage stats###
my-job-stats -a -n -s