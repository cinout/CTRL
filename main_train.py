import os
import argparse
import random
import torch.optim as optim
from datetime import datetime
from data_prepare.diffaugment import set_aug_diff, PoisonAgent
from methods import set_model
from methods.base import CLTrainer
from optimizer import LARS
from utils.util import *
from utils.frequency import PoisonFre
from utils.htba import PoisonHTBA
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from ssl_cleanse.ssl_cleanse import (
    trigger_inversion,
    trigger_mitigation,
)
import copy
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

parser = argparse.ArgumentParser(description="CTRL Training")

"""
Pretrained Models
"""
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


"""
Normalisation
"""
parser.add_argument(
    "--linear_probe_normalize",
    default="ref_set",
    choices=["regular", "none", "ref_set", "batch"],
)
parser.add_argument(
    "--normalize_backbone_features",
    default="none",
    choices=["l2", "none"],
)


"""
Image Size
"""
parser.add_argument("--window_size", default=32, type=int)
parser.add_argument(
    "--image_size", type=int, default=32
)  # 64 for imagenet-100, 32 for Cifar10/100

"""
Batch Size
"""
parser.add_argument("--eval_batch_size", default=512, type=int)
parser.add_argument("--linear_probe_batch_size", default=128, type=int)
parser.add_argument("--batch_size", default=128, type=int)


"""
Dataset
"""
parser.add_argument("--data_path", default="./datasets/")
parser.add_argument(
    "--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet100"]
)


"""
Architecture & Optimization
"""
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
parser.add_argument("--optimizer", default="sgd", choices=["sgd", "lars"])


"""
SSL
"""
parser.add_argument("--method", default="simclr", choices=["simclr", "byol", "mocov2"])
parser.add_argument("--temp", default=0.5, type=float)
parser.add_argument("--lr", default=0.06, type=float)
parser.add_argument("--wd", default=5e-4, type=float)
parser.add_argument("--cos", action="store_true", default=True)
parser.add_argument("--byol-m", default=0.996, type=float)

"""
Basics
"""
parser.add_argument("--note", type=str, default="")
parser.add_argument(
    "--timestamp",
    type=str,
    default=datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100)),
)
parser.add_argument("--ssl_pretrain_seed", default=42, type=int)
parser.add_argument("--num_workers", default=0, type=int)


"""
Epochs
"""
parser.add_argument("--epochs", default=800, type=int)
parser.add_argument("--frequency_detector_epochs", default=500, type=int)
parser.add_argument("--start_epoch", default=0, type=int)


"""
Logging & File Saving
"""
parser.add_argument(
    "--log_path", default="Experiments", type=str, help="path to save log"
)  # where checkpoints are stored
parser.add_argument("--saved_path", default="none", type=str)


"""
Distributed
"""
parser.add_argument("--distributed", action="store_true", help="distributed training")


"""
Evaluation
"""
parser.add_argument("--knn_eval_freq", default=5, type=int)


"""
Trigger / Poisoning
"""
parser.add_argument("--trigger_type", default="ftrojan", choices=["ftrojan", "htba"])
parser.add_argument("--target_class", default=0, type=int)
parser.add_argument("--poison_ratio", default=0.01, type=float)  # right value
parser.add_argument("--probe_set_percent", default=0.01, type=float)  # right value
parser.add_argument("--trigger_position", nargs="+", type=int, default=[15, 31])
parser.add_argument("--magnitude_train", default=50.0, type=float)  # right value
parser.add_argument("--magnitude_val", default=100.0, type=float)  # right value
parser.add_argument("--trigger_size", default=5, type=int)
parser.add_argument("--ftrojan_channel", nargs="+", type=int, default=[1, 2])


"""
Backdoor Detection Options
"""
parser.add_argument(
    "--bd_detectors",
    type=str,
    nargs="+",
    default=[],
    # choices=["entropy", "ss_score", "frequency_ensemble", "lid", "kdist"],
    help="applied detectors",
)
parser.add_argument(
    "--in_n_detectors",
    type=int,
    nargs="+",
    default=[1],
    help="the number of detectors the trigger index should be predicted in",
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

# Spectral Signature / Probe Dataset / Channel Detection, Removal, or Input Filtering
parser.add_argument(
    "--use_trigger_channel_removal",
    action="store_true",
    help="apply channel removal strategy",
)
parser.add_argument(
    "--remove_random_channels",
    action="store_true",
    help="a baseline: randomly drop out some channels",
)
parser.add_argument(
    "--remove_random_channels_seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--siftout_poisoned_images",
    action="store_true",
    help="use INPUT FILTERING",
)
parser.add_argument(
    "--replacement_value",
    type=str,
    choices=["zero", "ref_mean"],
    default="zero",
    help="determines what values to replace the old value at the trigger channels",
)
parser.add_argument(
    "--removed_channel_num",
    nargs="+",
    type=int,
    default=[2],
    help="remove k channels",
)
parser.add_argument(
    "--voted_channel_num",
    type=int,
    default=4,
    help="vote for k channels of EACH SAMPLE",
)
parser.add_argument(
    "--find_and_ignore_probe_channels",
    action="store_true",
    help="ignore channels from clean probe dataset",
)
parser.add_argument(
    "--ignore_probe_removed_channel_num",
    type=int,
    help="ignore those appear in the probe dataset's voted channels",
)
parser.add_argument(
    "--retrain_linear_after_channel_removal",
    action="store_true",
    help="allow re-training the linear classifier after the encoder is channel removed, using the 1 per cent ref set",
)
parser.add_argument(
    "--retrain_whole_model_after_cleanse",
    action="store_true",
    help="after cleanse method is applied, retrain the whole model (backbone+linear) using the 1 per cent ref set",
)
parser.add_argument(
    "--ideal_case",
    action="store_true",
    help="when we assume we have N real poisoned images",
)
parser.add_argument(
    "--knn_cluster_num",
    type=int,
    default=50,
    help="number of clusters",
)
parser.add_argument(
    "--find_channels_from_n_few_samples",
    type=int,
    default=0,
    help="If >0, sample from limited number of images for trigger channel",
)
parser.add_argument(
    "--match_with_clean_samples",
    type=int,
    default=0,
    help="Used together with find_channels_from_n_few_samples. If >0, sample some clean images as well",
)
parser.add_argument(
    "--use_complex_ss_aug",
    action="store_true",
    help="augment images for SS using more complex pipeline",
)
parser.add_argument(
    "--use_ss_contribute_percent",
    action="store_true",
    help="instead of using the frequency of appearance to estimate backdoor channels, use the total percentage of contribution to SS",
)
parser.add_argument(
    "--contribute_percent_option",
    type=str,
    choices=["standalone", "pick_from_voted"],
    default="standalone",
    help="(1) standalone: use as a stand-alone channel estimator; (2) pick_from_voted: from the pool of top channels voted, choose the ones with top contribution percent",
)
# parser.add_argument(
#     "--use_channel_var",
#     action="store_true",
#     help="use the channel's output variance as an indicator for backdoored channel",
# )
# parser.add_argument(
#     "--use_channel_var_option",
#     type=str,
#     choices=["union", "intersect"],
#     default="union",
#     help="for these channels, union or intersection with voted most frequent SS channels",
# )

# Frequency Detector
parser.add_argument(
    "--frequency_ensemble_size",
    type=int,
    default=1,
    help="the number of detectors in the frequency detector ensemble",
)
parser.add_argument(
    "--complex_gaussian",
    action="store_true",
)
parser.add_argument("--frequency_attack_trigger_ids", type=int, nargs="+", default=2)


# KDistance
parser.add_argument(
    "--kdist_k",
    type=int,
    default=8,
    help="distance the k-th neighbor",
)


"""
Image Augmentation
"""
parser.add_argument(
    "--num_views",
    type=int,
    default=1,
    help="how many views are generated for each image. Ultimately used by generate_view_tensors() function",
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


"""
Mask Pruning / Unlearning
"""
parser.add_argument(
    "--use_mask_pruning",
    action="store_true",
    help="apply mask pruning (RNP paper)",
)
parser.add_argument("--mask_pruning_seed", default=42, type=int)
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


"""
Competitor -- Defense Baseline: SSL Cleanse
"""
parser.add_argument(
    "--use_ssl_cleanse",
    action="store_true",
    help="use the method from ECCV2024 paper: ssl-cleanse",
)
parser.add_argument(
    "--ssl_cleanse_seed",
    type=int,
    default=10,
)
parser.add_argument(
    "--attack_succ_threshold",
    type=float,
    default=0.99,
    help="",
)
parser.add_argument(
    "--lam",
    type=float,
    default=0.1,
    help="",
)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lam_multiplier_up", type=float, default=1.5)
parser.add_argument("--ratio", type=float, default=0.05)
parser.add_argument(
    "--num_clusters",
    type=int,
    default=12,
)
parser.add_argument(
    "--trigger_path",
    default="",
    type=str,
)
parser.add_argument(
    "--mitigate_epochs",
    type=int,
    default=5,
)
parser.add_argument(
    "--draw_local_trigger_by", type=str, choices=["global", "local"], default="global"
)
parser.add_argument(
    "--drop",
    type=int,
    nargs="*",
    default=[50, 25],
    help="milestones for learning rate decay (0 = last epoch)",
)
parser.add_argument(
    "--drop_gamma",
    type=float,
    default=0.2,
    help="multiplicative factor of learning rate decay",
)
parser.add_argument("--eval_every", type=int, default=20, help="how often to evaluate")
parser.add_argument("--cj0", default=0.4, help="color jitter brightness")
parser.add_argument("--cj1", default=0.4, help="color jitter contrast")
parser.add_argument("--cj2", default=0.4, help="color jitter saturation")
parser.add_argument("--cj3", default=0.1, help="color jitter hue")
parser.add_argument("--cj_p", default=0.8, help="color jitter probability")
parser.add_argument("--gs_p", default=0.1, help="grayscale probability")
parser.add_argument("--crop_s0", default=0.2, help="crop size from")
parser.add_argument("--crop_s1", default=1.0, help="crop size to")
parser.add_argument("--crop_r0", default=0.75, help="crop ratio from")
parser.add_argument("--crop_r1", default=(4 / 3), help="crop ratio to")
parser.add_argument("--hf_p", default=0.5, help="horizontal flip probability")
parser.add_argument("--trigger_width", type=int, default=6)
parser.add_argument("--trigger_location", type=float, default=0.9)


"""
Baseline: MIMIC
"""
parser.add_argument(
    "--use_mimic",
    action="store_true",
    help="use the method from Mutual Information Guided Backdoor Mitigation for Pre-trained Encoders",
)
parser.add_argument("--mimic_seed", default=42, type=int)
parser.add_argument(
    "--mimic_lr", default=1e-2, type=float, help="initial learning rate"
)
parser.add_argument("--mimic_batch_size", default=128, type=int, help="")
parser.add_argument("--mimic_epochs", default=1000, type=int, help="")
parser.add_argument("--opt1", default=1000, type=int, help="opt1")
parser.add_argument("--opt2", default=1000, type=int, help="opt2")
parser.add_argument("--opt3", default=1000, type=int, help="opt3")
parser.add_argument("--opt4", default=1000, type=int, help="opt4")
parser.add_argument("--opt5", default=1, type=int, help="opt5")

"""
SSL Training Loss
"""
# FIXME: can remove
parser.add_argument(
    "--ssl_covariance_loss",
    action="store_true",
    help="add covariance loss regulariser for SSL training, idea from VICReg ICLR 2022 paper",
)
parser.add_argument(
    "--ssl_covariance_loss_w", type=float, default=1.0, help="coefficient of the loss"
)


"""
Others (Hopefully can be removed later)
"""
parser.add_argument(
    "--only_detect_projector_features",
    action="store_true",
    help="bd detectors use features from projector",
)


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    update_seed(args.ssl_pretrain_seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = (
        True  # forces cuDNN to use deterministic algorithms.
    )
    torch.backends.cudnn.benchmark = False  # avoids cuDNN choosing the fastest (but potentially nondeterministic) algorithm.

    """
    Create Model
    """
    print("=> creating cnn model '{}'".format(args.arch))

    # this is where model like simclr, byol is determined
    model = set_model(args)

    if args.pretrained_ssl_model != "":
        pretrained_state_dict = torch.load(
            args.pretrained_ssl_model, map_location=device
        )
        model.load_state_dict(pretrained_state_dict["state_dict"], strict=True)
    model = model.to(device)

    """
    Construct Trainer
    """
    trainer = CLTrainer(args)

    """
    Create Dataset/DataLoader
    """
    (
        train_dataset,
        test_dataset,
        memory_loader,
        train_transform,
    ) = set_aug_diff(args)

    """
    Create Poisoning Dataset
    """
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

    """
    Print All Args
    """
    all_args = "\n".join(
        "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
    )
    print(all_args)

    """
    Train and Evaluate
    """
    # update here
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
        )
    elif args.optimizer == "lars":
        optimizer = LARS(
            model.parameters(), args.lr, weight_decay=args.wd, momentum=0.9
        )

    # SSL attack and KNN Evaluation [Poisoned Model]
    trainer.train_freq(model, optimizer, train_transform, poison)

    # Linear Probe and Evaluation [Poisoned Model]
    backbone = extract_backbone(args.method, model)

    trained_linear = trainer.linear_probing(backbone, poison)

    """
    DEFENSE OPTIONS
    """

    """
    Ours: Channel Voting, Estimation, and Removal Strategy
    """
    if args.use_trigger_channel_removal:
        if args.find_channels_from_n_few_samples > 0:
            for _ in range(10):
                trainer.trigger_channel_removal(model, poison, trained_linear)
        else:
            trainer.trigger_channel_removal(model, poison, trained_linear)

    """
    Baseline 1: Use SSL-CLeanse (ECCV 2024 paper)
    """
    if args.use_ssl_cleanse:
        update_seed(args.ssl_cleanse_seed)

        backbone = extract_backbone(args.method, model)

        trainset_data = trigger_inversion(
            args, backbone, poison, model.feat_dim
        )  # trainset_data is a tuple of (x_untransformed, y)

        cleansed_backbone = trigger_mitigation(args, backbone, trainset_data)

        new_trainer = CLTrainer(args)

        clean_acc, back_acc = new_trainer.knn_monitor_fre(
            cleansed_backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            args,
            classes=args.num_classes,
            backdoor_loader=poison.test_pos_loader,
        )
        print(
            f">>>> With SSL-cleanse model, for kNN classifier, clean acc: {clean_acc:.1f}, back acc: {back_acc:.1f}",
        )

        _ = new_trainer.linear_probing(cleansed_backbone, poison, force_training=True)

    """
    Baseline 2: Mask Pruning Strategy
    """
    if args.use_mask_pruning:
        backbone = extract_backbone(args.method, model)
        update_seed(args.mask_pruning_seed)
        trainer.mask_prune(backbone, poison, trained_linear)

    """
    Baseline 3: Random Channel Removal, add args.remove_random_channels
    """

    """
    Baseline 4: MIMIC
    """
    if args.use_mimic:
        # teacher = extract_backbone(args.method, model)
        update_seed(args.mimic_seed)
        student = set_model(args)
        student = student.to(device)
        trainer.mimic(model, poison, student, train_transform)

        student.eval()
        for p in student.parameters():
            p.requires_grad = False

        if args.method == "mocov2":
            student_backbone = student.encoder_q
            student_backbone.fc = nn.Sequential()
        else:
            student_backbone = student.backbone

        # student_backbone = extract_backbone(args.method, student)

        # if args.method == "mocov2":
        #     # backbone = copy.deepcopy(model.encoder_q)
        #     student_backbone = type(student.encoder_q)()  # new instance
        #     student_backbone.load_state_dict(student.encoder_q.state_dict())
        #     student_backbone.fc = nn.Sequential()
        # else:
        #     # backbone = copy.deepcopy(model.backbone)
        #     student_backbone = type(student.backbone)()  # new instance
        #     student_backbone.load_state_dict(student.backbone.state_dict())

        new_trainer = CLTrainer(args)
        clean_acc, back_acc = new_trainer.knn_monitor_fre(
            student_backbone,
            poison.memory_loader,
            poison.test_clean_loader,
            args,
            classes=args.num_classes,
            backdoor_loader=poison.test_pos_loader,
        )
        print(
            f">>>> With MIMIC model, for kNN classifier, clean acc: {clean_acc:.1f}, back acc: {back_acc:.1f}",
        )
        _ = new_trainer.linear_probing(student_backbone, poison, force_training=True)

    """
    Other Cleanse Options: Input Filtering
    # FIXME: consider removing this option
    """
    # Sift out estimated poisoned images, and re-train the SSL model
    if args.siftout_poisoned_images:
        estimated_poisoned_file_indices = trainer.siftout_poisoned_images(
            model, poison, trained_linear
        )  # numpy

        print(
            f"estimated_poisoned_file_indices.shape: {estimated_poisoned_file_indices.shape}"
        )

        original_trainset_length = len(poison.train_pos_loader.dataset)
        estimated_clean_indices = np.setdiff1d(
            np.array(range(original_trainset_length)), estimated_poisoned_file_indices
        )

        poison.train_pos_loader = DataLoader(
            Subset(poison.train_pos_loader.dataset, estimated_clean_indices),
            batch_size=args.batch_size,
            sampler=None,
            shuffle=True,
            drop_last=False,
            # drop_last=True if args.method == "mocov2" else False,
        )
        print(f"filtered_dataset.shape: {len(poison.train_pos_loader.dataset)}")

        # re-train the model here
        new_model = set_model(args)
        new_model = new_model.to(device)
        new_trainer = CLTrainer(args)

        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
            )
        elif args.optimizer == "lars":
            optimizer = LARS(
                model.parameters(), args.lr, weight_decay=args.wd, momentum=0.9
            )

        # SSL attack and KNN Evaluation
        new_trainer.train_freq(
            new_model, optimizer, train_transform, poison, force_training=True
        )

        # Linear Probe and Evaluation
        backbone = extract_backbone(args.method, new_model)

        _ = new_trainer.linear_probing(backbone, poison, force_training=True)


if __name__ == "__main__":
    args = parser.parse_args()

    args.saved_path = os.path.join(
        f"./{args.log_path}/{args.timestamp}_{args.dataset}_{args.trigger_type}_{args.method}_{args.linear_probe_normalize}_sd{args.ssl_pretrain_seed}"
    )

    # Defense Baseline: SSL-Cleanse generated triggers
    if args.trigger_path == "":
        args.trigger_path = f"{args.timestamp}_trigger_estimation_{args.method}_{args.dataset}_{args.trigger_type}_SD{args.ssl_cleanse_seed}"

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    main(args)
