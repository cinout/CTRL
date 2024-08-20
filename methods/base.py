import os
import time
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data.distributed import DistributedSampler
from collections import Counter
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar
from utils.util import AverageMeter, save_model
from utils.knn import knn_monitor
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist

device = "cuda" if torch.cuda.is_available() else "cpu"


def find_trigger_channels(views, backbone, channel_num):
    # expected shaope of views: [bs, n_views, c, h, w]

    views = views.to(device)
    bs, n_views, c, h, w = views.shape
    views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
    vision_features = backbone(views)  # [bs*n_views, 512]
    _, c = vision_features.shape
    vision_features = vision_features.detach().cpu().numpy()
    u, s, v = np.linalg.svd(
        vision_features - np.mean(vision_features, axis=0, keepdims=True),
        full_matrices=False,
    )
    eig_for_indexing = v[0:1]  # [1, C]
    corrs = np.matmul(eig_for_indexing, np.transpose(vision_features))
    coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
    coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
    elementwise = (
        eig_for_indexing * vision_features * coeff_adjust
    )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign
    max_indices = np.argmax(elementwise, axis=1)
    occ_count = Counter(max_indices)
    essential_indices = occ_count.most_common(channel_num)
    print(
        f"essential_indices: {essential_indices}; #samples: {bs*n_views}"
    )  # print (idx, count) tuples
    essential_indices = torch.tensor(
        [idx for (idx, occ_count) in essential_indices]
    )  # remove count
    return essential_indices


# DISABLED, because it makes 0-channel_mean not 0, which is not good for our SS detecting strategy
def get_feats(loader, model, args):

    # switch to evaluate mode
    model.eval()
    feats, ptr = None, 0

    with torch.no_grad():
        for i, content in enumerate(loader):
            if args.detect_trigger_channels:
                (images, views, target, _) = content
            else:
                (images, target, _) = content

            # images = images.cuda(non_blocking=True)
            images = images.to(device)

            # Normalize for MoCo, BYOL etc.

            cur_feats = F.normalize(model(images), dim=1).cpu()  # default: L2 norm
            B, D = cur_feats.shape

            inds = torch.arange(B) + ptr  # [0, 1, ..., B-1] + prt

            if not ptr:
                # arrive only when ptr is 0 (i.e. first iteration)
                feats = torch.zeros(
                    (len(loader.dataset), D)
                ).float()  # len(loader.dataset) is the whole dataset's size, not just batch size

            # https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html
            feats.index_copy_(0, inds, cur_feats)  # (dim, index, tensor)

            ptr += B
    return feats


def train_linear_classifier(
    train_loader, backbone, linear, optimizer, args, use_ss_detector
):
    backbone.eval()
    linear.train()
    for i, content in enumerate(train_loader):

        if args.detect_trigger_channels:
            (images, views, target, _) = content
        else:
            (images, target, _) = content

        images = images.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = backbone(images)

            if args.detect_trigger_channels and use_ss_detector:
                # FIND channels that are related to trigger (although in training, all images are clean)
                essential_indices = find_trigger_channels(
                    views, backbone, args.channel_num
                )
                # set vallues to 0 at these indices
                output[:, essential_indices] = 0.0

        output = linear(output)
        loss = F.cross_entropy(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_linear_classifier(
    val_loader, backbone, linear, args, val_mode, use_ss_detector
):
    with torch.no_grad():
        acc1_accumulator = 0.0
        total_count = 0
        for i, content in enumerate(val_loader):
            if args.detect_trigger_channels:
                if val_mode == "poison":
                    (images, views, target, original_label, _) = content
                    original_label = original_label.to(device)
                elif val_mode == "clean":
                    (images, views, target, _) = content
                else:
                    raise Exception(f"unimplemented val_mode {val_mode}")
            else:
                if val_mode == "poison":
                    (images, target, original_label, _) = content
                    original_label = original_label.to(device)
                elif val_mode == "clean":
                    (images, target, _) = content
                else:
                    raise Exception(f"unimplemented val_mode {val_mode}")

            images = images.to(device)
            target = target.to(device)

            if val_mode == "poison":
                valid_indices = original_label != args.target_class
                if torch.all(~valid_indices):
                    # all inputs are from target class, skip this iteration
                    continue

                images = images[valid_indices]
                target = target[valid_indices]

            # compute output
            output = backbone(images)
            if args.detect_trigger_channels and use_ss_detector:
                # FIND channels that are related to trigger (although in training, all images are clean)
                essential_indices = find_trigger_channels(
                    views, backbone, args.channel_num
                )
                # set vallues to 0 at these indices
                output[:, essential_indices] = 0.0

            output = linear(output)
            _, pred = output.topk(
                1, 1, True, True
            )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
            # pred.shape: [bs, k=1]
            pred = pred.squeeze(1)  # shape: [bs, ]

            total_count += target.shape[0]
            acc1_accumulator += (pred == target).float().sum().item()

        return acc1_accumulator / total_count * 100.0


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer("inv_std", (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer("mean", mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class CLModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.method = args.method
        self.arch = args.arch
        self.dataset = args.dataset

        if "cifar" in self.dataset or "gtsrb" in self.dataset:
            print("CIFAR-variant Resnet is loaded")
            model_fun, feat_dim = model_dict_cifar[self.arch]
            self.mlp_layers = 2
        else:
            print("Original Resnet is loaded")
            model_fun, feat_dim = model_dict[self.arch]
            self.mlp_layers = 3

        self.model_generator = model_fun
        self.backbone = model_fun()
        # self.distill_backbone = model_fun()
        self.feat_dim = feat_dim

    def forward(self, x):
        pass

    def loss(self, reps):
        pass


class CLTrainer:
    def __init__(self, args):
        self.args = args
        # self.tb_logger = tb_logger.Logger(logdir=args.saved_path, flush_secs=2)
        self.tb_logger = SummaryWriter(log_dir=args.saved_path)
        logging.basicConfig(
            filename=os.path.join(self.tb_logger.log_dir, "training.log"),
            level=logging.DEBUG,
        )
        logging.info(str(args))

        self.args.warmup_epoch = 10

    def linear_probing(self, model, poison, use_ss_detector=False):
        linear_probing_epochs = 40
        if "cifar" in self.args.dataset or "gtsrb" in self.args.dataset:
            _, feat_dim = model_dict_cifar[self.args.arch]
        else:
            _, feat_dim = model_dict[self.args.arch]

        if self.args.method == "mocov2":
            backbone = (
                model.module.encoder_q if self.args.distributed else model.encoder_q
            )

            backbone.fc = nn.Sequential()
            for p in backbone.parameters():
                p.requires_grad = False
        else:
            backbone = model.backbone

        if self.args.use_ref_norm:
            train_probe_feats = get_feats(
                poison.train_probe_loader, backbone, self.args
            )
            train_var, train_mean = torch.var_mean(train_probe_feats, dim=0)

            linear = nn.Sequential(
                Normalize(),  # L2 norm
                FullBatchNorm(
                    train_var, train_mean
                ),  # the train_var/mean are from L2-normed features
                nn.Linear(feat_dim, self.args.num_classes),
            )
        else:
            linear = nn.Sequential(
                Normalize(),  # L2 norm
                nn.Linear(feat_dim, self.args.num_classes),
            )

        linear = linear.to(device)
        optimizer = torch.optim.SGD(
            linear.parameters(),
            lr=0.06,
            momentum=0.9,
            weight_decay=1e-4,
        )
        sched = [15, 30, 40]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched)

        # train linear classifier
        for epoch in range(linear_probing_epochs):
            print(f"training linear classifier, epoch: {epoch}")
            train_linear_classifier(
                poison.train_probe_loader,
                backbone,
                linear,
                optimizer,
                self.args,
                use_ss_detector=use_ss_detector,
            )
            # modify lr
            lr_scheduler.step()

        # eval linear classifier
        backbone.eval()
        linear.eval()

        print(f"evaluating linear classifier")
        print(f"evaluating on CLEAN val")
        clean_acc1 = eval_linear_classifier(
            poison.test_loader,
            backbone,
            linear,
            self.args,
            val_mode="clean",
            use_ss_detector=use_ss_detector,
        )
        print(f"evaluating on POISONED val")
        poison_acc1 = eval_linear_classifier(
            poison.test_pos_loader,
            backbone,
            linear,
            self.args,
            val_mode="poison",
            use_ss_detector=use_ss_detector,
        )
        print(
            f"with use_ss_detector set to: {use_ss_detector}, the ACC on clean val is: {clean_acc1}, the ASR on poisoned val is: {poison_acc1}"
        )

    # entry point of this file, called in main_train.py
    def train_freq(self, model, optimizer, train_transform, poison):

        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.epochs
        )
        warmup_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=self.args.warmup_epoch,
            after_scheduler=cosine_scheduler,
        )

        train_loader = poison.train_pos_loader
        test_loader = poison.test_loader  # clean val
        test_back_loader = poison.test_pos_loader  # poisoned val (test) set

        clean_acc = 0.0
        back_acc = 0.0

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                # calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
                train_loader.sampler.set_epoch(epoch)

            losses = AverageMeter()
            cl_losses = AverageMeter()

            train_transform = train_transform.to(device)

            # 1 epoch training
            start = time.time()

            # this is where training occurs
            for i, (images, __, _) in enumerate(
                train_loader
            ):  # frequency backdoor has been injected
                # print(i)
                model.train()
                images = images.to(device)

                # data
                v1 = train_transform(images)
                v2 = train_transform(images)

                if self.args.method == "simclr":
                    features = model(v1, v2)
                    loss, _, _ = model.criterion(features)

                elif self.args.method == "mocov2":
                    moco_losses = model(im_q=v1, im_k=v2)
                    loss = moco_losses.combine(
                        contr_w=1,
                        align_w=0,
                        unif_w=0,
                    )
                elif self.args.method == "simsiam":
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == "byol":
                    features = model(v1, v2)
                    loss = model.criterion(*features)

                elif self.args.method == "moco":

                    loss = model(v1, v2)

                losses.update(loss.item(), images[0].size(0))
                cl_losses.update(loss.item(), images[0].size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            warmup_scheduler.step()

            # (KNN-eval) why this eval step? (this code combines training and eval together)
            if epoch % self.args.knn_eval_freq == 0 or epoch + 1 == self.args.epochs:

                if self.args.method == "mocov2":
                    backbone = (
                        model.module.encoder_q
                        if self.args.distributed
                        else model.encoder_q
                    )
                    backbone.fc = nn.Sequential()
                    for p in backbone.parameters():
                        p.requires_grad = False
                else:
                    backbone = (
                        model.module.backbone
                        if self.args.distributed
                        else model.backbone
                    )

                clean_acc, back_acc = self.knn_monitor_fre(
                    backbone,
                    # model.module.backbone if self.args.distributed else model.backbone,
                    poison.memory_loader,  # memory loader is ONLY used here
                    test_loader,
                    epoch,
                    self.args,
                    classes=self.args.num_classes,
                    subset=False,
                    backdoor_loader=test_back_loader,
                )
                print(
                    "[{}-epoch] time:{:.3f} | clean acc: {:.3f} | back acc: {:.3f} | loss:{:.3f} | cl_loss:{:.3f}".format(
                        epoch + 1,
                        time.time() - start,
                        clean_acc,
                        back_acc,
                        losses.avg,
                        cl_losses.avg,
                    )
                )
                if epoch + 1 == self.args.epochs and self.args.detect_trigger_channels:
                    # if last epoch, also evaluate with SS detctor

                    if self.args.method == "mocov2":
                        backbone = (
                            model.module.encoder_q
                            if self.args.distributed
                            else model.encoder_q
                        )
                        backbone.fc = nn.Sequential()
                        for p in backbone.parameters():
                            p.requires_grad = False
                    else:
                        backbone = (
                            model.module.backbone
                            if self.args.distributed
                            else model.backbone
                        )

                    clean_acc_SSDETECTOR, back_acc_SSDETECTOR = self.knn_monitor_fre(
                        (
                            backbone
                            # model.module.backbone
                            # if self.args.distributed
                            # else model.backbone
                        ),
                        poison.memory_loader,  # memory loader is ONLY used here
                        test_loader,
                        epoch,
                        self.args,
                        classes=self.args.num_classes,
                        subset=False,
                        backdoor_loader=test_back_loader,
                        use_SS_detector=True,
                    )
                    print(
                        "[{}-epoch] time:{:.3f} | clean acc with SS Detector: {:.3f} | back acc with SS Detector: {:.3f} | loss:{:.3f} | cl_loss:{:.3f}".format(
                            epoch + 1,
                            time.time() - start,
                            clean_acc_SSDETECTOR,
                            back_acc_SSDETECTOR,
                            losses.avg,
                            cl_losses.avg,
                        )
                    )

        # Save final model
        if not self.args.distributed or (
            self.args.distributed and dist.get_rank() == 0
        ):
            save_model(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                filename=os.path.join(self.args.saved_path, "last.pth.tar"),
            )

            print("last epoch saved")

        return model

    @torch.no_grad()
    def knn_monitor_fre(
        self,
        net,
        memory_data_loader,
        test_data_loader,
        epoch,
        args,
        k=200,
        t=0.1,
        hide_progress=True,
        classes=-1,
        subset=False,
        backdoor_loader=None,
        use_SS_detector=False,
    ):

        net.eval()

        feature_bank = []
        # generate feature bank
        for data, target, _ in tqdm(
            memory_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            feature = net(data.to(device))

            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # feature_bank: [dim, total num]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()

        # feature_labels: [total num]
        feature_labels = (
            memory_data_loader.dataset[:][1].clone().detach().to(feature_bank.device)
        )

        """
        Evaluate clean KNN
        """

        clean_val_top1, clean_val_total_num = 0.0, 0
        test_bar = tqdm(test_data_loader, desc="kNN", disable=hide_progress)

        for content in test_bar:
            if args.detect_trigger_channels:
                (data, views, target, _) = content
            else:
                (data, target, _) = content

            data, target = data.to(device), target.to(device)
            feature = net(data)

            if use_SS_detector:
                essential_indices = find_trigger_channels(views, net, args.channel_num)
                feature[:, essential_indices] = 0.0

            feature = F.normalize(feature, dim=1)
            # feature: [bsz, dim]
            pred_labels = self.knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )

            clean_val_total_num += data.size(0)
            clean_val_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix(
                {"Accuracy": clean_val_top1 / clean_val_total_num * 100}
            )

        """
        Evaluate poison KNN
        """
        backdoor_val_top1, backdoor_val_total_num = 0.0, 0
        backdoor_test_bar = tqdm(backdoor_loader, desc="kNN", disable=hide_progress)

        for content in backdoor_test_bar:
            if args.detect_trigger_channels:
                (data, views, target, original_label, _) = content
            else:
                (data, target, original_label, _) = content

            data, target, original_label = (
                data.to(device),
                target.to(device),
                original_label.to(device),
            )

            valid_indices = original_label != args.target_class
            if torch.all(~valid_indices):
                # all inputs are from target class, skip this iteration
                continue

            data = data[valid_indices]
            target = target[valid_indices]

            feature = net(data)

            if use_SS_detector:
                essential_indices = find_trigger_channels(views, net, args.channel_num)
                feature[:, essential_indices] = 0.0

            feature = F.normalize(feature, dim=1)
            # feature: [bsz, dim]
            pred_labels = self.knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )

            backdoor_val_total_num += data.size(0)
            backdoor_val_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix(
                {"Accuracy": backdoor_val_top1 / backdoor_val_total_num * 100}
            )

        return (
            clean_val_top1 / clean_val_total_num * 100,
            backdoor_val_top1 / backdoor_val_total_num * 100,
        )

    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # feature: [bsz, dim]
        # feature_bank: [dim, clean_val_total_num]
        # feature_labels: [clean_val_total_num]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # sim_matrix: [bsz, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

        # sim_labels: [bsz, K]
        sim_labels = torch.gather(
            feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
        )
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(
            feature.size(0) * knn_k, classes, device=sim_labels.device
        )
        # one_hot_label: [bsz*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )  # for each row, only one column is 1, which is the label of k-nearest this neighbor

        # weighted score ---> [bsz, C]
        pred_scores = torch.sum(
            one_hot_label.view(feature.size(0), -1, classes)  # [bs, k, C=Classes]
            * sim_weight.unsqueeze(dim=-1),  # [bs, k, 1]
            dim=1,
        )  # [bs, C], where each column means the SCORE (weight) of the sample to the class at this column index

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels  # [bs, C], where the first column is the index (class) of nearest cluster
