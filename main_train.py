import os
import sys
import argparse
import warnings
import random

from utils.frequency import PoisonFre

import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import builtins

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from loaders.diffaugment import set_aug_diff, PoisonAgent
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description="CTRL Training")


parser.add_argument("--note", default="", type=str)
### dataloader
parser.add_argument("--data_path", default="./datasets/")
parser.add_argument(
    "--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet100"]
)
parser.add_argument("--full_dataset", action="store_true", default=True)
parser.add_argument("--window_size", default=32, type=int)
parser.add_argument("--image_size", default=32, type=int)  # 32 for CIFAR10/100
parser.add_argument("--eval_batch_size", default=512, type=int)
parser.add_argument("--linear_probe_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=1, type=int)


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

# ---- SLURM and DDP args ---- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--world_size",
    default=-1,
    type=int,
    help="number of processes for distributed training",
)
parser.add_argument(
    "--global_rank",
    default=-1,
    type=int,
    help="global rank for distributed training",
)
parser.add_argument(
    "--local_rank",
    default=-1,
    type=int,
    help="local rank for torch.distributed.launch training",
)
parser.add_argument(
    "--gpu",
    default=None,
    type=int,
    help="local rank for slurm training (c.f. local_rank)",
)
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist_backend", default="nccl", type=str, help="distributed backend"
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
parser.add_argument(
    "--use_ref_norm",
    action="store_true",
    help="normalize features by 1% trainset's mean and var",
)


# for finding trigger channels
parser.add_argument(
    "--detect_trigger_channels",
    action="store_true",
    help="use spectral signature to detect channels",
)
parser.add_argument(
    "--channel_num", default=1, type=int, help="number of channels to set to 0"
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


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


def main_worker(args):
    set_seed(args.seed)

    if not args.distributed or (args.distributed and dist.get_rank() == 0):
        args.timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(random.randint(0, 100))
            + "_"
            + str(random.randint(0, 100)),
        )

    args.saved_path = os.path.join(
        "./{}/{}-{}-{}-{}-poi{}-magtrain{}-magval{}-bs{}-lr{}-knnfreq{}-SSD{}-numc{}".format(
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
            args.channel_num,
        )
    )

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    # create model
    print("=> creating cnn model '{}'".format(args.arch))
    # this is where model like simclr, byol is determined
    model = set_model(args)

    if args.distributed:
        if args.gpu is None:
            model.cuda()
            model = DDP(model)
        else:
            model.cuda(args.gpu)
            model = DDP(model, device_ids=[args.gpu])

    # constrcut trainer
    trainer = CLTrainer(args)

    model = model.to(device)

    # create data loader
    (
        train_dataset,
        test_dataset,
        memory_loader,
        train_transform,
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

    # create optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    all_args = "\n".join(
        "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
    )
    print(all_args)

    # # Train

    # train a triggered model

    # model: simclr or byol
    # train_transform: augmentation for simclr/byol on the fly
    # poison: poisoned dataset, get train/test/memory via poison.xxx

    # actually, no need to return model, but it is also fine to return model
    model = trainer.train_freq(model, optimizer, train_transform, poison)

    if args.use_linear_probing:
        trainer.linear_probing(model, poison, use_ss_detector=False)
        if args.detect_trigger_channels:
            # comparison w. or w.o. SS Detector
            trainer.linear_probing(model, poison, use_ss_detector=True)


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.global_rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.global_rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.global_rank % ngpus_per_node
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.global_rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

        # suppress printing if not on master gpu
        if args.global_rank != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass
    main_worker()
