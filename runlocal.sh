python main_train.py \
    --method simclr \
    --threat_model our \
    --channel 1 2 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --lr 0.06 \
    --wd 0.0005 \
    --magnitude 100.0 \
    --poisoning \
    --epochs 800 \
    --gpu 0 \
    --window_size 32 \
    --trial test \
    --mode frequency \
    --dataset cifar10 \