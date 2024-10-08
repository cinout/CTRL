python -u main_train.py \
    --method mocov2 \
    --mode frequency \
    --channel 1 2 \
    --window_size 32 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --poisoning \
    \
    --batch_size 4 \
    --eval_batch_size 4 \
    --linear_probe_batch_size 4 \
    \
    --epochs 4 \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset imagenet100 \
    --target_class 26 \
    --image_size 32 \
    \
    --load_cached_tensors \
    # --use_linear_probing \
    # --detect_trigger_channels \
    # --channel_num 1 \
    # \
    # --note "try mocov2" \
    
    