python -u main_train.py \
    --method simclr \
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
    --epochs 5 \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    # --use_linear_probing \
    # --detect_trigger_channels \
    # --channel_num 1 \
