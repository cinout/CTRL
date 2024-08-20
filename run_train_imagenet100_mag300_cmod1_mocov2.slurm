#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg5|dlg6

#SBATCH --job-name="mocov2"
#SBATCH --account=punim1623
#SBATCH --time=1-08:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=80G

export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
export MASTER_PORT=28400
echo ">>> NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

srun python -u main_train.py \
    --method mocov2 \
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
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    --image_size 64 \
    \
    --use_linear_probing \
    --detect_trigger_channels \
    --channel_num 1 \
    \
    --note "try mocov2" \

##Log this job's resource usage stats###
my-job-stats -a -n -s