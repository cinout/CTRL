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
    --seed 42 \
    --num_views 64 \
    \
    --use_ssl_cleanse \
    --attack_succ_threshold 0.8 \
    --trigger_path trigger_estimation_DEBUG \
    --note "dehug" \