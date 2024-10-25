#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="c100_h"
#SBATCH --account=punim1623
#SBATCH --time=0-02:00:00

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
    --dataset cifar100 \
    --target_class 0 \
    \
    --pretrained_ssl_model "./Experiments/20240920_122148_13_42_cifar100_htba_linear_regular_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240924_154224_30_31_cifar100_htba_linear_refset_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    --pretrained_frequency_model "" \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --find_and_ignore_probe_channels \
    --ignore_probe_channel_num 30 \
    \
    --minority_lower_bound 0.001 \
    --minority_upper_bound 0.011 \
    \
    --bd_detectors kdist \
    --in_n_detectors 1 \
    --frequency_ensemble_size 3 \
    --frequency_detector_epochs 500 \
    --frequency_attack_trigger_ids 2 \
    \
    --seed 42 \
    \
    --num_views 1 \
    --normalize_backbone_features l2 \
    --kdist_k 8 \
    --note "[0.001-0.011] [1 KDist=8 detector, views=1] [encoder L2 norm]" \
##     --only_detect_projector_features \
##    --proj_feature_normalize l2 \
    ##    --compare_backbone_predictor \
    ##    --compare_mode default \
    ##    --normalize_backbone_features l2 \
        ##--only_detect_projector_features \


##Log this job's resource usage stats###
my-job-stats -a -n -s