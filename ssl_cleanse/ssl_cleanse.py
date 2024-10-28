import random
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


def trigger_inversion(args, backbone, poison, feat_dim):

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
        # x: [#total_images, 3, image_size, image_size], tensored (value range in 0-1), and transformed by above
        # x_untransformed: same shape as above, tensored (value range in 0-1), but no transformed
        # _ is gt label
        rep, x, x_untransformed, _ = get_data(
            device, backbone, dataloader, args.image_size, feat_dim, transform
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
            ).to(
                device
            )  # value range [-1, 1] -> arctanh -> (-inf, inf)
            delta = torch.arctanh(
                (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
            ).to(device)

            if args.trigger_set_number == 2:
                mask2 = torch.arctanh(
                    (torch.rand([1, 1, args.image_size, args.image_size]) - 0.5) * 2
                ).to(
                    device
                )  # value range [-1, 1] -> arctanh -> (-inf, inf)
                delta2 = torch.arctanh(
                    (torch.rand([1, 3, args.image_size, args.image_size]) - 0.5) * 2
                ).to(device)

            if args.use_dynamic_lam:
                mask_best = torch.tanh(mask) / 2 + 0.5
                delta_best = torch.tanh(delta) / 2 + 0.5
                if args.trigger_set_number == 2:
                    mask2_best = torch.tanh(mask2) / 2 + 0.5
                    delta2_best = torch.tanh(delta2) / 2 + 0.5

            mask.requires_grad = True
            delta.requires_grad = True
            if args.trigger_set_number == 2:
                mask2.requires_grad = True
                delta2.requires_grad = True

            if args.trigger_set_number == 1:
                opt = optim.Adam([delta, mask], lr=1e-1, betas=(0.5, 0.9))
            elif args.trigger_set_number == 2:
                opt = optim.Adam(
                    [delta, mask, delta2, mask2], lr=1e-1, betas=(0.5, 0.9)
                )

            if args.use_dynamic_lam:
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

                    mask_tanh = torch.tanh(mask) / 2 + 0.5  # value range (0, 1)
                    delta_tanh = torch.tanh(delta) / 2 + 0.5  # value range (0, 1)

                    X_R = draw(
                        images, args.mean, args.std, mask_tanh, delta_tanh
                    )  # draw trigger mask onto the image

                    loss_asr = norm_mse_loss(target_reps, backbone(X_R))
                    loss_reg = torch.mean(mask_tanh * delta_tanh)

                    if args.use_dynamic_lam:
                        loss = loss_asr + lam * loss_reg
                    else:
                        loss = loss_asr + args.lam * loss_reg

                    opt.zero_grad()
                    loss.backward(retain_graph=True)
                    opt.step()

                    # loss_asr_list.append(loss_asr.item())
                    loss_reg_list.append(loss_reg.item())
                    loss_list.append(loss.item())

                    if args.trigger_set_number == 2:
                        mask2_tanh = torch.tanh(mask2) / 2 + 0.5  # value range (0, 1)
                        delta2_tanh = torch.tanh(delta2) / 2 + 0.5  # value range (0, 1)
                        X_R = draw(images, args.mean, args.std, mask2_tanh, delta2_tanh)
                        loss_asr = norm_mse_loss(target_reps, backbone(X_R))
                        loss_reg = torch.mean(mask_tanh)

                        if args.use_dynamic_lam:
                            loss = loss_asr + lam * loss_reg
                        else:
                            loss = loss_asr + args.lam * loss_reg

                        opt.zero_grad()
                        loss.backward(retain_graph=True)
                        opt.step()

                        # loss_asr_list.append(loss_asr.item())
                        loss_reg_list.append(loss_reg.item())
                        loss_list.append(loss.item())

                # avg_loss_asr = torch.tensor(loss_asr_list).mean()
                avg_loss_reg = torch.tensor(loss_reg_list).mean()
                avg_loss = torch.tensor(loss_list).mean()

                """
                evaluate
                """
                # apply the learned trigger to all images
                if args.trigger_set_number == 1:
                    x_trigger = (
                        draw(x.to(device), args.mean, args.std, mask_tanh, delta_tanh)
                        .detach()
                        .to("cpu")
                    )
                elif args.trigger_set_number == 2:
                    x_trigger = (
                        draw(
                            x.to(device),
                            args.mean,
                            args.std,
                            mask_tanh,
                            delta_tanh,
                            mask2_tanh,
                            delta2_tanh,
                        )
                        .detach()
                        .to("cpu")
                    )

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
                    feat_dim,
                )

                print(f"ep: {ep}, asr_knn: {asr_knn:.3f}, avg_loss: {avg_loss:.3f}")

                if args.use_dynamic_lam:
                    if asr_knn > args.attack_succ_threshold and avg_loss_reg < reg_best:
                        mask_best = mask_tanh
                        delta_best = delta_tanh
                        reg_best = avg_loss_reg
                        if args.trigger_set_number == 2:
                            mask2_best = mask2_tanh
                            delta2_best = delta2_tanh
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
                else:
                    mask_best = mask_tanh
                    delta_best = delta_tanh
                    if args.trigger_set_number == 2:
                        mask2_best = mask2_tanh
                        delta2_best = delta2_tanh

            os.makedirs(args.trigger_path, exist_ok=True)
            if args.trigger_set_number == 1:
                torch.save(
                    {"mask": mask_best, "delta": delta_best},
                    os.path.join(args.trigger_path, f"{target}.pth"),
                )
            elif args.trigger_set_number == 2:
                torch.save(
                    {
                        "mask": mask_best,
                        "delta": delta_best,
                        "mask2": mask2_best,
                        "delta2": delta2_best,
                    },
                    os.path.join(args.trigger_path, f"{target}.pth"),
                )

    return (x_untransformed, y)


def trigger_mitigation(args, backbone, trainset_data):
    """
    setup frozen triggered encoder and learnable encoder
    """

    backbone_unlearn_trigger = copy.deepcopy(backbone)
    backbone_unlearn_trigger = backbone_unlearn_trigger.train()
    for param in backbone_unlearn_trigger.parameters():
        param.requires_grad = True

    backbone = backbone.eval()

    """
    set up optimizer and scheduler
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, backbone_unlearn_trigger.parameters()),
        lr=3e-3,
        weight_decay=1e-6,
    )
    scheduler = get_scheduler(args, optimizer)
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
    if args.trigger_set_number == 2:
        trigger_masks2 = []
        trigger_deltas2 = []

    for target in range(args.num_clusters):
        trigger_path = os.path.join(args.trigger_path, f"{target}.pth")
        trigger = torch.load(trigger_path, map_location=device)

        trigger_masks.append(trigger["mask"].detach())
        trigger_deltas.append(trigger["delta"].detach())
        if args.trigger_set_number == 2:
            trigger_masks2.append(trigger["mask2"].detach())
            trigger_deltas2.append(trigger["delta2"].detach())

    trigger_masks = torch.cat(trigger_masks, dim=0)
    trigger_deltas = torch.cat(trigger_deltas, dim=0)
    if args.trigger_set_number == 2:
        trigger_masks2 = torch.cat(trigger_masks2, dim=0)
        trigger_deltas2 = torch.cat(trigger_deltas2, dim=0)

    for ep in range(args.mitigate_epochs):

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
            if args.trigger_set_number == 2:
                mask2 = trigger_masks2[trigger_index]
                delta2 = trigger_deltas2[trigger_index]

            with torch.no_grad():
                clean_view_1_feature = backbone(clean_view_1)

            if random.random() < 0.5:
                compare_view = backbone_unlearn_trigger(clean_view_2)
            else:
                if args.trigger_overlay_option == 1:

                    if args.trigger_set_number == 1:
                        compare_view = backbone_unlearn_trigger(
                            draw(clean_view_3, args.mean, args.std, mask, delta)
                        )
                    elif args.trigger_set_number == 2:
                        if random.random() < 0.5:
                            compare_view = backbone_unlearn_trigger(
                                draw(clean_view_3, args.mean, args.std, mask, delta)
                            )
                        else:
                            compare_view = backbone_unlearn_trigger(
                                draw(clean_view_3, args.mean, args.std, mask2, delta2)
                            )

                elif args.trigger_overlay_option == 2:
                    trigger_width = random.randint(4, 10)

                    trigger_location_x = random.uniform(0.1, 0.9)
                    trigger_location_y = random.uniform(0.1, 0.9)

                    location_x = int(
                        (args.image_size - trigger_width) * trigger_location_x
                    )
                    location_y = int(
                        (args.image_size - trigger_width) * trigger_location_y
                    )

                    if args.trigger_set_number == 1:
                        applied_mask = mask
                        applied_delta = delta
                    elif args.trigger_set_number == 2:
                        if random.random() < 0.5:
                            applied_mask = mask
                            applied_delta = delta
                        else:
                            applied_mask = mask2
                            applied_delta = delta2

                    applied_mask = F.interpolate(
                        applied_mask, size=(trigger_width, trigger_width)
                    )
                    applied_delta = T.functional.normalize(
                        applied_delta, args.mean, args.std
                    )
                    applied_delta = F.interpolate(
                        applied_delta, size=(trigger_width, trigger_width)
                    )

                    clean_view_3[
                        :,
                        :,
                        location_x : location_x + trigger_width,
                        location_y : location_y + trigger_width,
                    ] = torch.mul(
                        clean_view_3[
                            :,
                            :,
                            location_x : location_x + trigger_width,
                            location_y : location_y + trigger_width,
                        ],
                        1 - applied_mask,
                    ) + torch.mul(
                        applied_delta, applied_mask
                    )

                    compare_view = backbone_unlearn_trigger(clean_view_3)
                elif args.trigger_overlay_option == 3:
                    trigger_width = random.randint(4, 10)

                    trigger_location_x = random.uniform(0.1, 0.9)
                    trigger_location_y = random.uniform(0.1, 0.9)

                    location_x = int(
                        (args.image_size - trigger_width) * trigger_location_x
                    )
                    location_y = int(
                        (args.image_size - trigger_width) * trigger_location_y
                    )

                    if args.trigger_set_number == 1:
                        applied_delta = delta
                    elif args.trigger_set_number == 2:
                        if random.random() < 0.5:
                            applied_delta = delta
                        else:
                            applied_delta = delta2

                    applied_delta = T.functional.normalize(
                        applied_delta, args.mean, args.std
                    )
                    applied_delta = F.interpolate(
                        applied_delta, size=(trigger_width, trigger_width)
                    )

                    clean_view_3[
                        :,
                        :,
                        location_x : location_x + trigger_width,
                        location_y : location_y + trigger_width,
                    ] = applied_delta

                    compare_view = backbone_unlearn_trigger(clean_view_3)

            loss_sum = norm_mse_loss(clean_view_1_feature, compare_view)

            loss_sum.backward()

            optimizer.step()

        scheduler.step()
        print(f"epoch {ep}, loss: {loss_sum.item()}")

    return backbone_unlearn_trigger
