python -u main_train.py \
    --mode frequency \
    --channel 1 2 \
    --method simclr \
    --window_size 32 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --poisoning \
    \
    --batch_size 4 \
    --eval_batch_size 4 \
    --linear_probe_batch_size 4 \
    \
    --epochs 800 \
    --magnitude_train 300.0 \
    --magnitude_val 300.0 \
    --dataset imagenet100 \
    --target_class 26 \
    --image_size 224 \
    \
    --use_linear_probing \
    --detect_trigger_channels \
    --channel_num 1 \
    \
    --note "size=224, second run, given more slurm time" \
    