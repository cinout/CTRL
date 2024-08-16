python main_train.py \
    --method simclr \
    --channel 1 2 \
    --trigger_position 15 31 \
    --poison_ratio 0.01 \
    --lr 0.06 \
    --wd 0.0005 \
    --poisoning \
    --window_size 32 \
    --mode frequency \
    --batch_size 4 \
    --epochs 5 \
    --dataset imagenet100 \
    --target_class 26 \
    # --dataset cifar10 \