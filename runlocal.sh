python -u main_train.py \
    --method simclr \
    --mode frequency \
    --channel 1 2 \
    --window_size 32 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --poisoning \
    \
    --batch_size 128 \
    --eval_batch_size 512 \
    --linear_probe_batch_size 128 \
    \
    --epochs 800 \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
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
    --pretrained_ssl_model "./Experiments/20240906_183247_61_43-cifar10-simclr-resnet18-poi0.01-magtrain50.0-magval100.0-bs512-lr0.06-knnfreq5-SSDYes/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240906_183247_61_43-cifar10-simclr-resnet18-poi0.01-magtrain50.0-magval100.0-bs512-lr0.06-knnfreq5-SSDYes/linear.pth.tar" \
    \
    --minority_criterion ss_score \
    \
    --use_frequency_detector \
    --frequency_detector_epochs 10 \
    \
    --note "use frequency detector" \