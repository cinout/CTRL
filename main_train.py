import os
import sys
import argparse
import warnings
import random


import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
from data_prepare.diffaugment import set_aug_diff, PoisonAgent


# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from utils.frequency import PoisonFre
from utils.htba import PoisonHTBA

parser = argparse.ArgumentParser(description="CTRL Training")

parser.add_argument("--trigger_type", default="ftrojan", choices=["ftrojan", "htba"])
parser.add_argument(
    "--pretrained_ssl_model",
    type=str,
    default="",
    help="path for pretrained ssl model (stage 1)",
)
parser.add_argument(
    "--pretrained_linear_model",
    type=str,
    default="",
    help="path for pretrained linear model (stage 2)",
)
parser.add_argument(
    "--pretrained_frequency_model",
    type=str,
    default="",
    help="path for pretrained frequency detector (stage 2)",
)
parser.add_argument(
    "--ignore_probe_channels",
    action="store_true",
    help="ignore channels from clean probe dataset",
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
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epochs", default=800, type=int)
parser.add_argument("--frequency_detector_epochs", default=500, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument(
    "--remove", default="none", choices=["crop", "flip", "color", "gray", "none"]
)
parser.add_argument("--update_model", action="store_true", default=False)
parser.add_argument("--contrastive", action="store_true", default=False)
parser.add_argument("--knn_eval_freq", default=5, type=int)
parser.add_argument("--distill_freq", default=5, type=int)
parser.add_argument("--saved_path", default="none", type=str)


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
parser.add_argument("--trigger_position", nargs="+", type=int, default=[15, 31])
parser.add_argument("--magnitude_train", default=50.0, type=float)  # right value
parser.add_argument("--magnitude_val", default=100.0, type=float)  # right value
parser.add_argument("--trigger_size", default=5, type=int)
parser.add_argument("--ftrojan_channel", nargs="+", type=int, default=[1, 2])
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


# parser.add_argument(
#     "--use_frequency_detector",
#     action="store_true",
#     help="use_frequency_detector to detect BD samples",
# )


# for finding trigger channels
parser.add_argument(
    "--detect_trigger_channels",
    action="store_true",
    help="use spectral signature to detect channels, this requires N augmented views to be generated",
)
parser.add_argument(
    "--bd_detectors",
    type=str,
    nargs="+",
    default=["frequency"],
    # choices=["entropy", "ss_score", "frequency"],
    help="applied detectors",
)


parser.add_argument(
    "--minority_lower_bound",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--minority_upper_bound",
    type=float,
    default=0.020,
)


# parser.add_argument(
#     "--minority_2nd_lower_bound",
#     type=float,
#     default=0.005,
# )
# parser.add_argument(
#     "--minority_2nd_upper_bound",
#     type=float,
#     default=0.500,
# )

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
    "--ignore_probe_channel_num",
    type=int,
    help="ignore those appear in the probe dataset's voted channels",
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


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):

    # os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.initial_seed()  # dataloader multi processing

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        # train_loader,
        # train_sampler,
        train_dataset,
        # ft_loader,
        # ft_sampler,
        # test_loader,
        test_dataset,
        memory_loader,
        train_transform,
        # ft_transform,
        # test_transform,
    ) = set_aug_diff(args)

    # create poisoning dataset
    if args.trigger_type == "ftrojan":
        poison_frequency_agent = PoisonFre(
            args,
            args.ftrojan_channel,
            args.window_size,
            args.trigger_position,
            False,
            True,
        )
    elif args.trigger_type == "htba":
        poison_frequency_agent = PoisonHTBA(
            args,
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

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )
    trainer.train_freq(model, optimizer, train_transform, poison)

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
    args = parser.parse_args()

    # for Logging
    if args.debug:  #### in the debug setting
        args.saved_path = os.path.join("./{}/test".format(args.log_path))
    else:
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
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    main(args)
