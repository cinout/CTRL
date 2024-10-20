import os
import argparse
import random
import torch.optim as optim
from datetime import datetime
from data_prepare.diffaugment import set_aug_diff, PoisonAgent
from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from utils.frequency import PoisonFre
from utils.htba import PoisonHTBA
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from ssl_cleanse import (
    DatasetEval,
    DatasetInit,
    dataloader_cluster,
    draw,
    eval_knn,
    get_data,
    norm_mse_loss,
)
import copy
import torch.nn as nn
import torchvision.transforms as T

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
    "--find_and_ignore_probe_channels",
    action="store_true",
    help="ignore channels from clean probe dataset",
)

parser.add_argument("--note", type=str, default="")
parser.add_argument("--image_size", type=int, default=32)


parser.add_argument(
    "--linear_probe_normalize",
    default="regular",
    choices=["regular", "none", "ref_set", "batch"],
)
parser.add_argument(
    "--normalize_backbone_features",
    default="none",
    choices=["l2", "none"],
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
    default=["frequency_ensemble"],
    # choices=["entropy", "ss_score", "frequency_ensemble", "lid", "kdist"],
    help="applied detectors",
)
parser.add_argument(
    "--frequency_ensemble_size",
    type=int,
    default=1,
    help="the number of detectors in the frequency detector ensemble",
)
# parser.add_argument(
#     "--frequency_train_trigger_size",
#     type=int,
#     default=2,
#     help="the number of triggers to choose from for training frequency detector",
# )
parser.add_argument(
    "--in_n_detectors",
    type=int,
    nargs="+",
    default=[1],
    help="the number of detectors the trigger index should be predicted in",
)
parser.add_argument("--frequency_attack_trigger_ids", type=int, nargs="+", default=2)
parser.add_argument(
    "--complex_gaussian",
    action="store_true",
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
parser.add_argument(
    "--unlearn_before_finding_trigger_channels",
    action="store_true",
    help="unlearn the model before finding trigger channels",
)

parser.add_argument(
    "--ideal_case",
    action="store_true",
    help="when trigger channels are found from val set directly",
)
parser.add_argument(
    "--full_dataset_svd",
    action="store_true",
    help="apply spectral signature on whole dataset",
)
parser.add_argument(
    "--knn_before_svd",
    action="store_true",
    help="apply kNN to features before performing spectral signature",
)
parser.add_argument(
    "--knn_cluster_num",
    type=int,
    default=50,
    help="number of clusters",
)
parser.add_argument(
    "--siftout_poisoned_images",
    action="store_true",
    help="use input filtering",
)

parser.add_argument(
    "--only_detect_projector_features",
    action="store_true",
    help="bd detectors use features from projector",
)
parser.add_argument(
    "--compare_backbone_predictor",
    action="store_true",
    help="use the difference of BD detector's scores between backbone and predictor to predict bd samples",
)
parser.add_argument(
    "--compare_mode",
    type=str,
    choices=["default", "abs"],
    default="default",
    help="how to compare",
)
parser.add_argument(
    "--proj_feature_normalize",
    default="none",
    choices=["none", "l2"],
)

# KDistance
parser.add_argument(
    "--kdist_k",
    type=int,
    default=32,
    help="distance the k-th neighbor",
)

# TODO: add to slurm
parser.add_argument(
    "--use_ssl_cleanse",
    action="store_true",
    help="use the method from ECCV2024 paper: ssl-cleanse",
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
    default=1e-1,
    help="",
)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--lam_multiplier_up", type=float, default=1.5)
parser.add_argument("--ratio", type=float, default=0.05)
parser.add_argument("--knn_sample_num", type=int, default=1000, required=True)
parser.add_argument("--num_clusters", type=int, default=12, required=True)

# TODO: add to slurm
parser.add_argument("--trigger_path", type=str, required=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    # SSL attack and KNN Evaluation
    trainer.train_freq(model, optimizer, train_transform, poison)

    # Linear Probe and Evaluation
    trained_linear = trainer.linear_probing(model, poison)

    if args.use_ssl_cleanse:
        if args.method == "mocov2":
            backbone = copy.deepcopy(model.encoder_q)
            backbone.fc = nn.Sequential()
        else:
            backbone = copy.deepcopy(model.backbone)

        backbone = backbone.eval()

        with torch.no_grad():
            transform = T.Compose(
                [
                    # T.Resize(args.image_size),
                    # T.ToTensor(),
                    T.Normalize(args.mean, args.std),
                ]
            )
            dataloader = DataLoader(
                dataset=DatasetInit(poison.train_probe_loader, transform, args.ratio),
                batch_size=100,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            rep, x, y_true = get_data(
                device,
                backbone,
                dataloader,
                args.image_size,
                model.feat_dim,
            )
            kmeans = KMeans(
                n_clusters=args.num_clusters, random_state=0, n_init=30
            ).fit(rep)
            y = kmeans.labels_
            counts_label = {}  # # of images belonging to cluster i
            for i in range(np.unique(y).shape[0]):
                mask = y == i
                counts_label[i] = mask.sum()  # #images belonging to cluster i

            rep_center = torch.empty(
                (len(np.unique(y)), rep.shape[1])
            )  # [#labels, rep_dim]
            y_center = torch.empty(len(np.unique(y)))  # [#labels,]
            for label in np.unique(y):
                rep_center[label, :] = rep[y == label].mean(dim=0)
                y_center[label] = label
            rep_knn, y_knn = rep, torch.tensor(y)

        reg_best_list = torch.empty(len(np.unique(y)))  # [#labels,]
        loss_f = norm_mse_loss

        for target in np.unique(y):
            rep_target = rep[y == target]  # [#images_in_target_cluster, rep_dim]
            x_other = x[y != target]  # [#other_images]
            x_other_indices = torch.randperm(x_other.shape[0])[
                : x.shape[0] - max(counts_label.values())
            ]
            x_other_sample = x_other[x_other_indices]

            mask = torch.arctanh(
                (torch.rand([1, 1, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)

            delta = torch.arctanh(
                (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)

            mask.requires_grad = True
            delta.requires_grad = True
            opt = optim.Adam([delta, mask], lr=1e-1, betas=(0.5, 0.9))

            reg_best = torch.inf

            lam = 0
            cost_set_counter = 0
            cost_up_counter = 0
            cost_down_counter = 0

            dataloader_train = dataloader_cluster(args, rep_target, x_other_sample, 32)

            for ep in range(1000):
                loss_asr_list, loss_reg_list, loss_list = [], [], []
                for n_iter, (images, target_reps) in enumerate(dataloader_train):
                    images = images.to(device)
                    target_reps = target_reps.to(device)
                    mask_tanh = torch.tanh(mask) / 2 + 0.5
                    delta_tanh = torch.tanh(delta) / 2 + 0.5
                    X_R = draw(images, args.mean, args.std, mask_tanh, delta_tanh)
                    z = target_reps
                    zt = backbone(X_R)
                    loss_asr = loss_f(z, zt)
                    loss_reg = torch.mean(mask_tanh * delta_tanh)
                    loss = loss_asr + lam * loss_reg
                    opt.zero_grad()
                    loss.backward(retain_graph=True)
                    opt.step()

                    loss_asr_list.append(loss_asr.item())
                    loss_reg_list.append(loss_reg.item())
                    loss_list.append(loss.item())

                avg_loss_asr = torch.tensor(loss_asr_list).mean()
                avg_loss_reg = torch.tensor(loss_reg_list).mean()
                avg_loss = torch.tensor(loss_list).mean()

                x_trigger = (
                    draw(x.to(device), args.mean, args.std, mask_tanh, delta_tanh)
                    .detach()
                    .to("cpu")
                )

                dataloader_eval = DataLoader(
                    dataset=DatasetEval(x_trigger, args.knn_sample_num),
                    batch_size=100,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )

                # dataloader_eval = ds.dataloader_knn(x_trigger, args.knn_sample_num)
                asr_knn = eval_knn(
                    device,
                    backbone,
                    dataloader_eval,
                    rep_knn,
                    y_knn,
                    target,
                    model.feat_dim,
                )

                if asr_knn > args.attack_succ_threshold and avg_loss_reg < reg_best:
                    mask_best = mask_tanh
                    delta_best = delta_tanh
                    reg_best = avg_loss_reg

                print(
                    "step: %3d, lam: %.2E, asr: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f"
                    % (ep, lam, asr_knn, avg_loss, avg_loss_asr, avg_loss_reg, reg_best)
                )

                if lam == 0 and asr_knn >= args.attack_succ_threshold:
                    cost_set_counter += 1
                    if cost_set_counter >= args.patience:
                        lam = args.lam
                        cost_up_counter = 0
                        cost_down_counter = 0
                        # logging.info("initialize cost to %.2E" % lam)
                else:
                    cost_set_counter = 0

                if asr_knn >= args.attack_succ_threshold:
                    cost_up_counter += 1
                    cost_down_counter = 0
                else:
                    cost_up_counter = 0
                    cost_down_counter += 1

                if lam != 0 and cost_up_counter >= args.patience:
                    cost_up_counter = 0
                    # logging.info(
                    #     "up cost from %.2E to %.2E"
                    #     % (lam, lam * args.lam_multiplier_up)
                    # )
                    lam *= args.lam_multiplier_up

                elif lam != 0 and cost_down_counter >= args.patience:
                    cost_down_counter = 0
                    # logging.info(
                    #     "down cost from %.2E to %.2E"
                    #     % (lam, lam / args.lam_multiplier_up)
                    # )
                    lam /= args.lam_multiplier_up

            reg_best_list[target] = reg_best if reg_best != torch.inf else 1

            os.makedirs(args.trigger_path, exist_ok=True)
            torch.save(
                {"mask": mask_best, "delta": delta_best},
                os.path.join(args.trigger_path, f"{target}.pth"),
            )

        return

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
        )
        print(f"filtered_dataset.shape: {len(poison.train_pos_loader.dataset)}")

        # re-train the model here
        new_model = set_model(args)
        new_model = new_model.to(device)
        new_trainer = CLTrainer(args)
        optimizer = optim.SGD(
            new_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
        )
        # SSL attack and KNN Evaluation
        new_trainer.train_freq(
            new_model, optimizer, train_transform, poison, force_training=True
        )

        # Linear Probe and Evaluation
        new_trained_linear = new_trainer.linear_probing(
            new_model, poison, force_training=True
        )

        return  # we can exit now

    # Channel Removal Strategy
    if args.detect_trigger_channels:
        trainer.trigger_channel_removal(model, poison, trained_linear)

    # Mask Pruning Strategy
    if args.use_mask_pruning:
        trainer.linear_probing(
            model, poison, use_mask_pruning=True, trained_linear=trained_linear
        )


if __name__ == "__main__":
    args = parser.parse_args()

    args.saved_path = os.path.join(
        f"./{args.log_path}/{args.timestamp}_{args.dataset}_{args.trigger_type}_linear_{args.linear_probe_normalize}_sd{args.seed}_[RAW]"
    )
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    main(args)
