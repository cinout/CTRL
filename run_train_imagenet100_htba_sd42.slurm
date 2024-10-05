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
#SBATCH --time=0-05:30:00

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
    --trigger_type htba \
    \
    --image_size 64 \
    \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    \
    --pretrained_ssl_model "./Experiments/20240920_122142_53_0_imagenet100_htba_linear_regular_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240924_155156_28_85_imagenet100_htba_linear_refset_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    --pretrained_frequency_model "" \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --ignore_probe_channels \
    --ignore_probe_channel_num 30 \
    \
    --minority_lower_bound 0.003 \
    --minority_upper_bound 0.050 \
    \
    --bd_detectors frequency_ensemble \
    --frequency_ensemble_size 3 \
    --frequency_train_trigger_size 5 \
    --in_n_detectors 3 \
    --frequency_detector_epochs 500 \
    --attack_trigger_ids 2 \
    --complex_gaussian \
    \
    --seed 42 \
    \
    --note "attack_trigger_ids=2, [0.3-5.0]%, EP=500, [More Complex GaussNoise]" \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s