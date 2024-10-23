import torch.nn.functional as F
from torch.utils import data
import torch
import copy
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
import os
from ssl_cleanse.inversion import (
    DatasetEval,
    DatasetInit,
    dataloader_cluster,
    draw,
    eval_knn,
    get_data,
    # norm_mse_loss,
)
from ssl_cleanse.mitigation import ds_train, get_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def trigger_inversion(args, model, poison):
    if args.method == "mocov2":
        backbone = copy.deepcopy(model.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model.backbone)

    backbone = backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    with torch.no_grad():
        """
        prepare dataset
        """

        dataloader = DataLoader(
            dataset=DatasetInit(poison.train_probe_loader),
            batch_size=100,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        transform = T.Compose(
            [
                T.Normalize(args.mean, args.std),
            ]
        )
        # rep: [#total_images, feat_dim]
        # x: [#total_images, 3, image_size, image_size], transformed by above
        # x_untransformed: same shape as above, but no transformed
        # _ is gt label
        rep, x, x_untransformed, _ = get_data(
            device, backbone, dataloader, args.image_size, model.feat_dim, transform
        )

        """
        Clustering
        """
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=30).fit(
            rep
        )
        y = kmeans.labels_  # predicted cluster ids

        counts_label = {}  # # of images belonging to cluster i
        for i in range(np.unique(y).shape[0]):
            mask = y == i
            counts_label[i] = mask.sum()  # #images belonging to cluster i

        # rep_center = torch.empty(
        #     (len(np.unique(y)), rep.shape[1])
        # )  # [#clusters, feat_dim], averaged over all images in the cluster

        # for label in np.unique(y):
        #     rep_center[label, :] = rep[y == label].mean(dim=0)

        # rep_knn, y_knn = rep, torch.tensor(y)

    # reg_best_list = torch.empty(len(np.unique(y)))  # [#clusters,]

    # estimate trigger for each cluster
    for target in np.unique(y):  # for each cluster

        if not os.path.exists(
            os.path.join(args.trigger_path, f"{target}.pth")
        ):  # if trigger is not available yet
            """
            set up data of target cluster and other clusters
            """
            rep_target = rep[y == target]  # [#target_cluster_size, rep_dim]
            x_other = x[y != target]  # [#other_images, 3, image_size, image_size]
            x_other_indices = torch.randperm(x_other.shape[0])[
                : x.shape[0] - max(counts_label.values())
            ]
            x_other_sample = x_other[
                x_other_indices
            ]  # other clusters' images in a shuffled order, [~#other_images, 3, image_size, image_size]

            """
            initialize mask and delta
            """
            mask = torch.arctanh(
                (torch.rand([1, 1, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)
            delta = torch.arctanh(
                (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)
            mask.requires_grad = True
            delta.requires_grad = True
            opt = optim.Adam([delta, mask], lr=1e-1, betas=(0.5, 0.9))

            reg_best = (
                torch.inf
            )  # records the current best (smallest) regression loss (constraining the size and magnitude of triggers)
            lam = 0  # coefficient for two losses
            cost_set_counter = 0
            cost_up_counter = 0
            cost_down_counter = 0

            dataloader_train = dataloader_cluster(args, rep_target, x_other_sample)

            for ep in range(1000):
                """
                train and learn triggers
                """
                loss_asr_list, loss_reg_list, loss_list = [], [], []
                for images, target_reps in dataloader_train:

                    images = images.to(device)  # image from another cluster
                    target_reps = target_reps.to(
                        device
                    )  # target cluster image representation

                    mask_tanh = torch.tanh(mask) / 2 + 0.5
                    delta_tanh = torch.tanh(delta) / 2 + 0.5
                    X_R = draw(
                        images, args.mean, args.std, mask_tanh, delta_tanh
                    )  # draw trigger mask onto the image

                    loss_asr = norm_mse_loss(target_reps, backbone(X_R))
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

                """
                evaluate
                """
                x_trigger = (
                    draw(x.to(device), args.mean, args.std, mask_tanh, delta_tanh)
                    .detach()
                    .to("cpu")
                )  # apply the learned trigger to all images

                # shuffle, and pick 1000 images
                dataloader_eval = DataLoader(
                    dataset=DatasetEval(x_trigger, 1000),
                    batch_size=100,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )

                # return the percentage of triggered images that are predictd to be the current cluster, aka, attack success rate
                asr_knn = eval_knn(
                    device,
                    backbone,
                    dataloader_eval,
                    rep,  # ALL clean images' latent representation
                    torch.tensor(y),  # ALL predicted cluster ids
                    target,  # current cluster id
                    model.feat_dim,
                )

                if asr_knn > args.attack_succ_threshold and avg_loss_reg < reg_best:
                    # update the optimal mask and delta
                    mask_best = mask_tanh
                    delta_best = delta_tanh
                    reg_best = avg_loss_reg

                print(
                    "step: %3d, lam: %.2E, asr: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f"
                    % (ep, lam, asr_knn, avg_loss, avg_loss_asr, avg_loss_reg, reg_best)
                )

                """
                adjusting lambda
                """
                if lam == 0 and asr_knn >= args.attack_succ_threshold:
                    cost_set_counter += 1
                    if cost_set_counter >= args.patience:  # >=5 patience is 5
                        lam = args.lam  # reset lambda to initial value
                        cost_up_counter = 0
                        cost_down_counter = 0
                else:
                    cost_set_counter = 0

                if asr_knn >= args.attack_succ_threshold:
                    cost_up_counter += 1
                    cost_down_counter = 0
                else:
                    cost_up_counter = 0
                    cost_down_counter += 1

                if lam != 0 and cost_up_counter >= args.patience:
                    # boost up lambda
                    cost_up_counter = 0
                    lam *= args.lam_multiplier_up

                elif lam != 0 and cost_down_counter >= args.patience:
                    # bring down lambda
                    cost_down_counter = 0
                    lam /= args.lam_multiplier_up

            # reg_best_list[target] = reg_best if reg_best != torch.inf else 1

            os.makedirs(args.trigger_path, exist_ok=True)
            torch.save(
                {"mask": mask_best, "delta": delta_best},
                os.path.join(args.trigger_path, f"{target}.pth"),
            )

    return (x_untransformed, y)


def trigger_mitigation(args, model, trainset_data):
    """
    setup frozen triggered encoder and learnable encoder
    """
    if args.method == "mocov2":
        backbone = copy.deepcopy(model.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model.backbone)

    backbone_unlearn_trigger = copy.deepcopy(backbone)
    backbone_unlearn_trigger = backbone_unlearn_trigger.train()
    for param in backbone_unlearn_trigger.parameters():
        param.requires_grad = True

    backbone = backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    """
    set up optimizer and scheduler
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, backbone_unlearn_trigger.parameters()),
        lr=3e-3,
        weight_decay=1e-6,
    )
    scheduler = get_scheduler(args, optimizer)
    eval_every = args.eval_every
    lr_warmup = 0
    torch.backends.cudnn.benchmark = True

    """
    setup dataloader
    """
    dataloader = DataLoader(
        dataset=ds_train(args, trainset_data),
        batch_size=128,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trigger_masks = []
    trigger_deltas = []
    for target in range(args.num_clusters):
        trigger_path = os.path.join(args.trigger_path, f"{target}.pth")
        trigger = torch.load(trigger_path, map_location=device)

        trigger_masks.append(trigger["mask"].detach())
        trigger_deltas.append(trigger["delta"].detach())
    trigger_masks = torch.cat(trigger_masks, dim=0)
    trigger_deltas = torch.cat(trigger_deltas, dim=0)

    for ep in range(args.mitigate_epoches):

        for clean_view_1, clean_view_2, clean_view_3, trigger_index in dataloader:
            clean_view_1 = clean_view_1.to(device)  # [bs, 3, img_size, img_size]
            clean_view_2 = clean_view_2.to(device)  # [bs, 3, img_size, img_size]
            clean_view_3 = clean_view_3.to(device)  # [bs, 3, img_size, img_size]
            trigger_index = trigger_index.to(device)  # [bs]

            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = 3e-3 * lr_scale
                lr_warmup += 1
            optimizer.zero_grad()

            mask = trigger_masks[trigger_index]  # [bs, 1, img_size, img_size]
            delta = trigger_deltas[trigger_index]  # [bs, 3, img_size, img_size]

            poison_view = torch.mul(clean_view_3, 1 - mask) + torch.mul(
                delta, mask
            )  # [bs, 3, img_size, img_size]

            clean_view_1_feature = backbone(clean_view_1)
            clean_view_2_feature = backbone_unlearn_trigger(clean_view_2)
            poison_view_feature = backbone_unlearn_trigger(poison_view)

            loss_1 = norm_mse_loss(clean_view_1_feature, clean_view_2_feature)
            loss_2 = norm_mse_loss(clean_view_1_feature, poison_view_feature)

            loss_sum = loss_1 + loss_2

            loss_sum.backward()

            optimizer.step()

        scheduler.step()
        print(f"epoch {ep}, loss: {loss_sum.item()}")

    return backbone_unlearn_trigger
