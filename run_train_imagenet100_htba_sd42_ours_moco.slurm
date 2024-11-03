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
#SBATCH --time=0-09:00:00

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
    --pretrained_ssl_model "./Experiments/20241024_235233_77_100_imagenet100_htba_linear_ref_set_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20241103_191643_57_83_imagenet100_htba_linear_ref_set_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    \
    --seed 42 \
    --num_views 64 \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    --find_and_ignore_probe_channels \
    --ignore_probe_channel_num 30 \
    \
    --minority_lower_bound 0.000 \
    --minority_upper_bound 0.010 \
    \
    --bd_detectors ss_score \
    --normalize_backbone_features none \
    --in_n_detectors 1 \
    \
    --note "[channel=40, detector=ss_score]" \
    --trigger_path "20241030_173710_75_75_trigger_estimation_mocov2_imagenet100_htba" \
    --num_clusters 100 \
##    --retrain_linear_after_channel_removal \
## \
##    --find_and_ignore_probe_channels \
##    --ignore_probe_channel_num 30 \

    

##Log this job's resource usage stats###
my-job-stats -a -n -s