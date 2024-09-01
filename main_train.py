import os
import sys
import argparse
import warnings
import random

from utils.frequency import PoisonFre

import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from loaders.diffaugment import set_aug_diff, PoisonAgent

parser = argparse.ArgumentParser(description="CTRL Training")


parser.add_argument(
    "--pretrained_ssl_model",
    type=str,
    default="",
    help="path for pretrained ssl model (stage 1)",
)
parser.add_argument("--note", type=str, default="")
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument(
    "--use_ref_norm",
    action="store_true",
    help="normalize features by 1% trainset's mean and var",
)

### dataloader
parser.add_argument("--data_path", default="./datasets/")
parser.add_argument(
    "--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet100"]
)
parser.add_argument("--disable_normalize", action="store_true", default=True)
parser.add_argument("--full_dataset", action="store_true", default=True)
parser.add_argument("--window_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=512, type=int)
parser.add_argument("--linear_probe_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=1, type=int)

parser.add_argument(
    "--timestamp",
    type=str,
    default=datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100)),
)


### training
parser.add_argument(
    "--arch",
    default="resnet18",
    type=str,
    choices=[
        "resnet18",
        "resnet50",
        "resnet101",
        "shufflenet",
        "mobilenet",
        "squeezenet",
    ],
)
parser.add_argument("--method", default="simclr", choices=["simclr", "byol", "mocov2"])
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--epochs", default=800, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument(
    "--remove", default="none", choices=["crop", "flip", "color", "gray", "none"]
)
parser.add_argument("--poisoning", action="store_true", default=False)
parser.add_argument("--update_model", action="store_true", default=False)
parser.add_argument("--contrastive", action="store_true", default=False)
parser.add_argument("--knn_eval_freq", default=5, type=int)
parser.add_argument("--distill_freq", default=5, type=int)
parser.add_argument("--saved_path", default="none", type=str)
parser.add_argument("--mode", default="normal", choices=["normal", "frequency"])


## ssl setting
parser.add_argument("--temp", default=0.5, type=float)
parser.add_argument("--lr", default=0.06, type=float)
parser.add_argument("--wd", default=5e-4, type=float)
parser.add_argument("--cos", action="store_true", default=True)
parser.add_argument("--byol-m", default=0.996, type=float)


###poisoning
parser.add_argument("--target_class", default=0, type=int)
parser.add_argument("--poison_ratio", default=0.01, type=float)  # right value
parser.add_argument("--pin_memory", action="store_true", default=False)
parser.add_argument("--reverse", action="store_true", default=False)
parser.add_argument("--trigger_position", nargs="+", type=int)
parser.add_argument("--magnitude_train", default=50.0, type=float)  # right value
parser.add_argument("--magnitude_val", default=100.0, type=float)  # right value
parser.add_argument("--trigger_size", default=5, type=int)
parser.add_argument("--channel", nargs="+", type=int)
parser.add_argument("--loss_alpha", default=2.0, type=float)
parser.add_argument("--strength", default=1.0, type=float)  ### augmentation strength

### for convenient debugging
parser.add_argument(
    "--load_cached_tensors",
    action="store_true",
)

###logging
parser.add_argument(
    "--log_path", default="Experiments", type=str, help="path to save log"
)  # where checkpoints are stored
parser.add_argument("--debug", action="store_true", default=False)

###others
parser.add_argument("--distributed", action="store_true", help="distributed training")
parser.add_argument("--seed", default=42, type=int)

### linear probing
parser.add_argument(
    "--use_linear_probing",
    action="store_true",
    help="evaluate the performance using linear probing",
)


# for finding trigger channels
parser.add_argument(
    "--detect_trigger_channels",
    action="store_true",
    help="use spectral signature to detect channels, this requires N augmented views to be generated",
)
parser.add_argument(
    "--minority_percent",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--replacement_value",
    type=str,
    choices=["zero", "ref_mean"],
    default="zero",
    help="determines what values to replace the old value at the trigger channels",
)
parser.add_argument(
    "--channel_num",
    nargs="+",
    type=int,
    help="a new hp, determine k channels of EACH SAMPLE",
)
parser.add_argument(
    "--num_views",
    type=int,
    default=64,
    help="how many views are generated for each image, for NeighborVariation detector",
)
parser.add_argument(
    "--rrc_scale_min",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--rrc_scale_max",
    type=float,
    default=0.95,
)

# for mask pruning
parser.add_argument(
    "--use_mask_pruning",
    action="store_true",
    help="apply mask pruning (RNP paper)",
)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument(
    "--clean_threshold",
    type=float,
    default=0.20,
    help="threshold of unlearning accuracy",
)
parser.add_argument(
    "--unlearning_lr",
    type=float,
    default=0.01,
    help="the learning rate for neuron unlearning",
)
parser.add_argument(
    "--recovering_lr",
    type=float,
    default=0.2,
    help="the learning rate for mask optimization",
)
parser.add_argument(
    "--unlearning_epochs",
    type=int,
    default=20,
    help="the number of epochs for unlearning",
)
parser.add_argument(
    "--recovering_epochs",
    type=int,
    default=20,
    help="the number of epochs for recovering",
)
parser.add_argument(
    "--pruning-by", type=str, default="threshold", choices=["number", "threshold"]
)
parser.add_argument(
    "--pruning-max",
    type=float,
    default=0.90,
    help="the maximum number/threshold for pruning",
)
parser.add_argument(
    "--pruning-step",
    type=float,
    default=0.05,
    help="the step size for evaluating the pruning",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[10, 20],
    help="Decrease learning rate at these epochs.",
)


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

# for Logging
if args.debug:  #### in the debug setting
    args.saved_path = os.path.join("./{}/test".format(args.log_path))
else:
    if args.mode == "normal":
        args.saved_path = os.path.join(
            "./{}/{}-{}_{}".format(
                args.log_path,
                args.dataset,
                args.method,
                args.arch,
            )
        )
    elif args.mode == "frequency":
        # poisoning
        args.saved_path = os.path.join(
            "./{}/{}-{}-{}-{}-poi{}-magtrain{}-magval{}-bs{}-lr{}-knnfreq{}-SSD{}".format(
                args.log_path,
                args.timestamp,
                args.dataset,
                args.method,
                args.arch,
                args.poison_ratio,
                args.magnitude_train,
                args.magnitude_val,
                args.batch_size,
                args.lr,
                args.knn_eval_freq,
                "Yes" if args.detect_trigger_channels else "No",
            )
        )
    else:
        raise Exception(f"args.mode {args.mode} is not implemented")


if not os.path.exists(args.saved_path):
    os.makedirs(args.saved_path)

# tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)


def main():
    print(args.saved_path)
    set_seed(args.seed)

    main_worker(args)


def main_worker(args):

    # create model
    print("=> creating cnn model '{}'".format(args.arch))
    # this is where model like simclr, byol is determined
    model = set_model(args)
    if args.pretrained_ssl_model != "":
        pretrained_state_dict = torch.load(
            args.pretrained_ssl_model, map_location=device
        )
        model.load_state_dict(pretrained_state_dict["state_dict"], strict=True)

    model = model.to(device)

    # constrcut trainer
    trainer = CLTrainer(args)

    # create data loader
    (
        train_loader,
        train_sampler,
        train_dataset,
        ft_loader,
        ft_sampler,
        test_loader,
        test_dataset,
        memory_loader,
        train_transform,
        ft_transform,
        test_transform,
    ) = set_aug_diff(args)

    # create poisoning dataset
    if args.poisoning:
        poison_frequency_agent = PoisonFre(
            args,
            args.channel,
            args.window_size,
            args.trigger_position,
            False,
            True,
        )
        poison = PoisonAgent(
            args,
            poison_frequency_agent,
            train_dataset,
            test_dataset,
            memory_loader,
            args.magnitude_train,
            args.magnitude_val,
        )

    all_args = "\n".join(
        "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
    )
    print(all_args)

    ###### train a triggered model
    # model: simclr or byol
    # train_transform: augmentation for simclr/byol on the fly
    # poison: poisoned dataset, get train/test/memory via poison.xxx

    if args.pretrained_ssl_model == "":
        # create optimizer
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
        )
        trainer.train_freq(model, optimizer, train_transform, poison)

    if args.use_linear_probing:
        trained_linear = trainer.linear_probing(model, poison, use_ss_detector=False)

        if args.detect_trigger_channels:
            # comparison w. or w.o. SS Detector
            trainer.linear_probing(
                model, poison, use_ss_detector=True, trained_linear=trained_linear
            )

        if args.use_mask_pruning:
            trainer.linear_probing(
                model, poison, use_mask_pruning=True, trained_linear=trained_linear
            )


if __name__ == "__main__":
    main()
