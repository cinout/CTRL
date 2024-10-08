python -u main_train.py \
    --method simclr \
    \
    --magnitude_train 50.0 \
    --magnitude_val 100.0 \
    --dataset cifar10 \
    --target_class 0 \
    \
    --pretrained_ssl_model "./Experiments/20240906_183247_61_43_cifar10_ftrojan_linear_regular_sd42/last.pth.tar" \
    --pretrained_linear_model "./Experiments/20240924_154021_3_40_cifar10_ftrojan_linear_refset_sd42/linear.pth.tar" \
    --linear_probe_normalize ref_set \
    --pretrained_frequency_model "" \
    \
    --detect_trigger_channels \
    --channel_num 40 \
    \
    --find_and_ignore_probe_channels \
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
    \
    --seed 42 \
    \
    --note "" \