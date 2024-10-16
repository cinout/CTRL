import copy
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import roc_auc_score
from collections import Counter
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar
from utils.util import AverageMeter, save_model
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
from networks.mask_batchnorm import MaskBatchNorm2d
import h5py
import PIL
import random
from frequency_detector import FrequencyDetector, patching_train, dct2
from methods.maskprune import (
    test_maskprune,
    evaluate_by_threshold,
    read_data,
    save_mask_scores,
    refill_unlearned_model,
    train_step_recovering,
    train_step_unlearning,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_freq_detection_scores(images, freq_detector_ensemble, bd_detector_scores, args):
    if "frequency_ensemble" in args.bd_detectors:
        # evaluate
        images = torch.permute(images, (0, 2, 3, 1))
        images = np.array(
            images.cpu(), dtype=np.float32
        )  # shape: [bs, 32, 32, 3]; value range: [0, 1]
        for i in range(images.shape[0]):
            for channel in range(3):
                images[i][:, :, channel] = dct2(
                    (images[i][:, :, channel] * 255).astype(np.uint8)
                )
        images = torch.tensor(images, device=device)
        images = torch.permute(images, (0, 3, 1, 2))  # shape: [bs, 3, 32, 32]
        for ensemble_id in range(args.frequency_ensemble_size):
            freq_detector = freq_detector_ensemble[ensemble_id]
            freq_detector.eval()
            output = freq_detector(
                images
            )  # [bs, 2], the second element is anomaly score
            output = output[:, 1].detach().cpu().tolist()
            bd_detector_scores[f"frequency_ensemble_{ensemble_id}"].extend(output)


def lid_mle(data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    b = data.shape[0]
    k = min(k, b - 2)

    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)

    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.0e-4), dim=1)
    return lids


def get_pairwise_distance(
    data, reference, compute_mode="use_mm_for_euclid_dist_if_necessary"
):
    b = data.shape[0]
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    # offset = misc.get_rank() * b
    offset = 0
    mask = torch.zeros((b, reference.shape[0]), device=data.device, dtype=torch.bool)
    mask = torch.diagonal_scatter(
        mask, torch.ones(b), offset
    )  # False, with diagonal value (distance to itself) set to True
    r = r[~mask].view(b, -1)
    return r  # [b, b-1]


def get_detection_scores(
    vision_features,
    corrs,
    max_indices_at_channel,
    bd_detector_scores,
    args,
):
    if "entropy" in args.bd_detectors:
        for votes in max_indices_at_channel:  # for each original image
            votes_counter = Counter(votes).most_common()
            counts = np.array([c for (name, c) in votes_counter])
            p = counts / counts.sum()
            h = -np.sum(p * np.log(p))
            entropy = -1 * np.exp(h)
            bd_detector_scores["entropy"].append(entropy)

    if "ss_score" in args.bd_detectors:
        corrs = np.abs(corrs)
        corrs = corrs.reshape(-1, args.num_views)  #  [bs,n_views]
        ss_scores = np.max(corrs, axis=1)  # [bs]
        bd_detector_scores["ss_score"].extend(ss_scores.tolist())

    if "lid" in args.bd_detectors:
        lids = lid_mle(
            data=vision_features.detach(), reference=vision_features.detach()
        )
        lids = lids.reshape(-1, args.num_views)  #  [bs,n_views]
        lids = torch.mean(lids, dim=1)
        bd_detector_scores["lid"].extend(lids.cpu().numpy())

    if "kdist" in args.bd_detectors:
        d = get_pairwise_distance(
            vision_features.detach(),
            vision_features.detach(),
        )
        a, _ = torch.sort(d, dim=1)
        a = a[:, 16]  # TODO: may need to upscale for n_views=64
        a = a.reshape(-1, args.num_views)  #  [bs,n_views]
        a = torch.mean(a, dim=1)
        bd_detector_scores["kdist"].extend(a.cpu().numpy())


def get_detection_scores_from_projector(
    vision_features, projector, bs, bd_detector_scores, args
):
    vision_features = projector(vision_features)
    if args.proj_feature_normalize == "l2":
        vision_features = F.normalize(vision_features, dim=1)
    _, C = vision_features.shape
    corrs, max_indices_at_channel = get_ss_statistics(
        vision_features.detach().cpu().numpy(), bs, C, args
    )
    get_detection_scores(
        vision_features, corrs, max_indices_at_channel, bd_detector_scores, args
    )


def ss_statistics(visual_features, bs, feat_dim, args, probe_set=False):
    u, s, v = np.linalg.svd(
        visual_features - np.mean(visual_features, axis=0, keepdims=True),
        full_matrices=False,
    )

    # get top eigenvector
    eig_for_indexing = v[0:1]  # [1, C]

    # adjust direction (sign)
    corrs = np.matmul(eig_for_indexing, np.transpose(visual_features))  # [1, bs*n_view]

    coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
    coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
    elementwise = (
        eig_for_indexing * visual_features * coeff_adjust
    )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign

    # get contributing indices sorted from low to high
    max_indices = np.argsort(
        elementwise, axis=1
    )  # [bs*n_view, C], C are indices, sorted by value from low to high

    max_indices = max_indices.reshape(bs, args.num_views, feat_dim)  # [bs, n_view, C]

    if probe_set:
        take_channel = args.ignore_probe_channel_num
    else:
        take_channel = max(args.channel_num)

    max_indices_at_channel = max_indices[
        :, :, -take_channel:
    ]  # [bs, n_view, take_channel]
    max_indices_at_channel = max_indices_at_channel.reshape(
        bs, -1
    )  # [bs, n_view*take_channel]

    return corrs, max_indices_at_channel


def get_ss_statistics(
    visual_features, bs, feat_dim, args, probe_set=False, is_poisoned=None
):
    # is_poisoned is the GTs for poisoned train set, when not None, means the function is called by train set

    if args.knn_before_svd:
        if is_poisoned:
            gt = torch.cat(is_poisoned)
            gt = np.array(gt.cpu())  # [#dataset]

        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        # iso = IsolationForest(contamination=0.05)
        # y_iso = iso.fit_predict(visual_features)
        # X_filtered = visual_features[y_iso == 1]

        # scaler = RobustScaler()
        # pca = PCA(n_components=2)

        # clusters = KMeans(
        #     n_clusters=args.knn_cluster_num, n_init="auto", init="k-means++"
        # ).fit(pca.fit_transform(visual_features))
        # labels = clusters.labels_

        # dbscan = DBSCAN(eps=0.5, min_samples=5)
        # labels = dbscan.fit_predict(scaler.fit_transform(visual_features))

        gmm = GaussianMixture(n_components=args.knn_cluster_num, random_state=42)
        labels = gmm.fit_predict(scaler.fit_transform(visual_features))
        # num_classes = set(labels)

        corrs_total = np.zeros(shape=(1, bs), dtype=visual_features.dtype)
        if probe_set:
            take_channel = args.ignore_probe_channel_num
        else:
            take_channel = max(args.channel_num)
        max_indices_at_channel_total = np.zeros(
            shape=(bs, take_channel), dtype=np.int64
        )

        for cluster_id in set(labels):  # FIXME: update
            matching_indices = labels == cluster_id  # An array of True and False

            if is_poisoned:
                total_poisoned_in_cluster = gt[matching_indices].sum()
                print(
                    f">>>> [TrainSet] in cluster {cluster_id}, #total: {np.nonzero(matching_indices)[0].shape[0]}, #poisoned: {total_poisoned_in_cluster}"
                )
            else:
                print(
                    f">>>> [ProbeSet] in cluster {cluster_id}, #total: {np.nonzero(matching_indices)[0].shape[0]}"
                )

            cluster_features = visual_features[matching_indices]
            corrs, max_indices_at_channel = ss_statistics(
                cluster_features, cluster_features.shape[0], feat_dim, args, probe_set
            )

            # need to remember the indices of the statistics
            corrs_total[:, matching_indices] = corrs
            max_indices_at_channel_total[matching_indices, :] = max_indices_at_channel

        # print(
        #     f"max_indices_at_channel_total.dtype: {max_indices_at_channel_total.dtype}"
        # )
        # print(
        #     f"max_indices_at_channel_total.shape: {max_indices_at_channel_total.shape}"
        # )
        # print(f"corrs_total.dtype: {corrs_total.dtype}")
        # print(f"corrs_total.shape: {corrs_total.shape}")

        return corrs_total, max_indices_at_channel_total
    else:
        return ss_statistics(visual_features, bs, feat_dim, args, probe_set)


def generate_view_tensors(input, ss_transform):
    # input.shape: [total, 3, 32, 32]; value range: [0, 1]
    input = torch.permute(input, (0, 2, 3, 1))
    input = input * 255.0
    input = torch.clamp(input, 0, 255)
    input = np.array(
        input.cpu(), dtype=np.uint8
    )  # shape: [total, 32, 32, 3]; value range: [0, 255]

    view_tensors = []
    for img in input:
        img = PIL.Image.fromarray(img)  # in PIL format now
        views = ss_transform(
            img
        )  # a list of args.num_views elements, each one is a PIL image

        tensors_of_an_image = []
        for view in views:
            view = np.asarray(view).astype(np.float32) / 255.0
            view = torch.tensor(view)
            view = torch.permute(view, (2, 0, 1))  # shape: [c=3, h, w], value: [0, 1]
            tensors_of_an_image.append(view)
        tensors_of_an_image = torch.stack(
            tensors_of_an_image, dim=0
        )  # [num_views, c, h, w]
        view_tensors.append(tensors_of_an_image)

    view_tensors = torch.stack(view_tensors, dim=0)  # [total, num_views, c, h, w]

    return view_tensors


def find_trigger_channels(
    args,
    data_loader,  # train dataset with poisoned images
    train_probe_loader,
    train_probe_freq_detector_loader,
    backbone,
    projector,
    linear,
    ss_transform,
):
    bd_detector_scores = dict()
    for detector in args.bd_detectors:
        if detector == "frequency_ensemble":
            for i in range(args.frequency_ensemble_size):
                bd_detector_scores[f"{detector}_{i}"] = []
        else:
            bd_detector_scores[detector] = []

    all_votes = []  # for all images in the dataset
    is_poisoned = []  # for all images in the dataset (GT)
    if args.siftout_poisoned_images:
        trainset_file_indices = []

    # if apply unlearning before finding trigger channles
    # NOT USED
    if args.unlearn_before_finding_trigger_channels:
        unlearnt_backbone = copy.deepcopy(backbone)
        unlearnt_linear = copy.deepcopy(linear)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            list(unlearnt_backbone.parameters()) + list(unlearnt_linear.parameters()),
            lr=args.unlearning_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=0.1
        )
        for epoch in range(0, args.unlearning_epochs + 1):
            train_acc = train_step_unlearning(
                args=args,
                model=unlearnt_backbone,
                linear=unlearnt_linear,
                criterion=criterion,
                optimizer=optimizer,
                data_loader=train_probe_loader,
            )

            scheduler.step()
            print(f">>>>>>>> at epoch {epoch}, the train_acc is {train_acc}")

            if train_acc <= args.clean_threshold:
                print(f">>>>>>>> arrive at early break of unlearning at epoch {epoch}")
                break
        unlearnt_backbone.eval()
        unlearnt_linear.eval()

    # to train frequency detectors
    if "frequency_ensemble" in args.bd_detectors:
        freq_detector_ensemble = []
        for ensemble_id in range(args.frequency_ensemble_size):
            freq_detector = FrequencyDetector(
                height=args.image_size, width=args.image_size
            )
            freq_detector = freq_detector.to(device)
            if args.pretrained_frequency_model == "":
                # train from scratch
                optimizer = torch.optim.Adadelta(
                    freq_detector.parameters(), lr=0.05, weight_decay=1e-4
                )
                criterion = nn.CrossEntropyLoss()
                freq_detector.train()

                for epoch in range(args.frequency_detector_epochs):
                    for content in train_probe_freq_detector_loader:
                        # prepare data in this batch
                        (images_clean, _, _) = content
                        images_clean = images_clean.to(device)
                        images_clean = torch.permute(images_clean, (0, 2, 3, 1))
                        images_clean = np.array(
                            images_clean.cpu(), dtype=np.float32
                        )  # shape: [bs, 32, 32, 3]; value range: [0, 1]
                        images_poi = np.zeros_like(images_clean)
                        for i in range(images_clean.shape[0]):
                            images_poi[i] = patching_train(
                                images_clean[i],
                                images_clean,
                                args.image_size,
                                ensemble_id,
                                args.frequency_attack_trigger_ids,
                                args.complex_gaussian,
                            )

                        images = np.concatenate(
                            [images_clean, images_poi], axis=0
                        )  # shape: [2*bs, 32, 32, 3]; value range: [0, 1]
                        for i in range(images.shape[0]):
                            for channel in range(3):
                                images[i][:, :, channel] = dct2(
                                    (images[i][:, :, channel] * 255).astype(np.uint8)
                                )
                        labels = np.concatenate(
                            (
                                np.zeros(images_clean.shape[0]),
                                np.ones(images_clean.shape[0]),
                            ),
                            axis=0,
                        )

                        idx = np.arange(images.shape[0])
                        random.shuffle(idx)
                        images = images[
                            idx
                        ]  # shape: [2*bs, 32, 32, 3]; value range: [0, 1]
                        images = torch.tensor(images, device=device)
                        images = torch.permute(
                            images, (0, 3, 1, 2)
                        )  # shape: [2*bs, 3, 32, 32]

                        labels = labels[idx]  # shape: [2*bs]
                        labels = torch.tensor(labels, device=device, dtype=torch.long)

                        # obtain loss and update params
                        output = freq_detector(images)  # [2*bs, 2]
                        loss = criterion(output, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    print(f"> epoch is {epoch}; loss is {loss.item()}")
                save_model(
                    freq_detector.state_dict(),
                    filename=os.path.join(
                        args.saved_path, f"frequency_ensemble_{ensemble_id}.pth.tar"
                    ),
                )
            else:
                # load model
                pretrained_state_dict = torch.load(
                    f"{args.pretrained_frequency_model}_{ensemble_id}.pth.tar",
                    map_location=device,
                )
                freq_detector.load_state_dict(pretrained_state_dict, strict=True)
            freq_detector_ensemble.append(freq_detector)

    if args.full_dataset_svd:
        h5py_filename = (
            f"{args.timestamp}_{args.dataset}_{args.trigger_type}_features.hdf5"
        )
        h5py_handler = h5py.File(h5py_filename, "w")

        if "cifar" in args.dataset or "gtsrb" in args.dataset:
            _, feat_dim = model_dict_cifar[args.arch]
        else:
            _, feat_dim = model_dict[args.arch]

    """
    # called if we want to ignore some clean channels voted by train_probe dataset
    """
    if args.find_and_ignore_probe_channels:
        all_probe_votes = []

        if args.full_dataset_svd:
            train_probe_set_features = h5py_handler.create_dataset(
                "train_probe_set_features",
                (len(train_probe_loader.dataset) * args.num_views, feat_dim),
            )
            start_pos = 0
            for i, content in enumerate(train_probe_loader):
                (images, _, _) = content

                images = images.to(device)

                if args.num_views == 1:
                    views = images.clone()
                    views = views.unsqueeze(1)
                else:
                    views = generate_view_tensors(images, ss_transform)

                views = views.to(device)

                bs, n_views, c, h, w = views.shape
                views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
                if args.unlearn_before_finding_trigger_channels:
                    vision_features = unlearnt_backbone(views)
                else:
                    vision_features = backbone(views)  # [bs*n_views, 512]

                if args.normalize_backbone_features == "l2":
                    vision_features = F.normalize(vision_features, dim=-1)
                _, C = vision_features.shape
                vision_features = vision_features.detach().cpu().numpy()
                train_probe_set_features[start_pos : vision_features.shape[0]] = (
                    vision_features
                )
                start_pos = start_pos + vision_features.shape[0]

            # print(f"train_probe_set_features.shape: {train_probe_set_features.shape}")
            corrs, max_indices_at_channel = get_ss_statistics(
                train_probe_set_features,
                int(train_probe_set_features.shape[0] / args.num_views),
                train_probe_set_features.shape[1],
                args,
                probe_set=True,
            )
            all_probe_votes.append(max_indices_at_channel)
        else:
            for i, content in enumerate(train_probe_loader):
                (images, _, _) = content

                images = images.to(device)

                if args.num_views == 1:
                    views = images.clone()
                    views = views.unsqueeze(1)
                else:
                    views = generate_view_tensors(images, ss_transform)

                views = views.to(device)

                bs, n_views, c, h, w = views.shape
                views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
                if args.unlearn_before_finding_trigger_channels:
                    vision_features = unlearnt_backbone(views)
                else:
                    vision_features = backbone(views)  # [bs*n_views, 512]

                if args.normalize_backbone_features == "l2":
                    vision_features = F.normalize(vision_features, dim=-1)
                _, C = vision_features.shape
                vision_features = vision_features.detach().cpu().numpy()

                corrs, max_indices_at_channel = get_ss_statistics(
                    vision_features, bs, C, args, probe_set=True
                )
                all_probe_votes.append(max_indices_at_channel)

    """
    Actual train loader with 1% poisoned images
    """

    if args.full_dataset_svd:
        trainset_features = h5py_handler.create_dataset(
            "trainset_features", (len(data_loader.dataset) * args.num_views, feat_dim)
        )
        start_pos = 0
        for i, content in tqdm(enumerate(data_loader)):
            if args.ideal_case:
                images = content[0]
                is_batch_poisoned = torch.ones(size=(images.shape[0],))
                is_batch_poisoned = is_batch_poisoned.to(device)
            else:
                (images, is_batch_poisoned, _, _) = content
                is_batch_poisoned = is_batch_poisoned.to(device)

            images = images.to(device)

            if args.num_views == 1:
                views = images.clone()
                views = views.unsqueeze(1)
            else:
                views = generate_view_tensors(images, ss_transform)
            views = views.to(device)

            bs, n_views, c, h, w = views.shape
            views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
            if args.unlearn_before_finding_trigger_channels:
                vision_features = unlearnt_backbone(views)
            else:
                vision_features = backbone(views)  # [bs*n_views, 512]

            if args.normalize_backbone_features == "l2":
                vision_features = F.normalize(vision_features, dim=-1)
            _, C = vision_features.shape
            vision_features = vision_features.detach().cpu().numpy()
            trainset_features[start_pos : vision_features.shape[0]] = vision_features
            start_pos = start_pos + vision_features.shape[0]
            is_poisoned.append(is_batch_poisoned)
            if "frequency_ensemble" in args.bd_detectors:
                get_freq_detection_scores(
                    images, freq_detector_ensemble, bd_detector_scores, args
                )

        # print(f"trainset_features.shape: {trainset_features.shape}")
        corrs, max_indices_at_channel = get_ss_statistics(
            trainset_features,
            int(trainset_features.shape[0] / args.num_views),
            trainset_features.shape[1],
            args,
            is_poisoned=is_poisoned,
        )
        if args.detect_projector_features:
            get_detection_scores_from_projector(
                trainset_features,
                projector,
                int(trainset_features.shape[0] / args.num_views),
                bd_detector_scores,
                args,
            )
            pass
        else:
            get_detection_scores(
                trainset_features,
                corrs,
                max_indices_at_channel,
                bd_detector_scores,
                args,
            )

        all_votes.append(max_indices_at_channel)

    else:
        for i, content in tqdm(enumerate(data_loader)):
            if args.ideal_case:
                images = content[0]
                is_batch_poisoned = torch.ones(size=(images.shape[0],))
                is_batch_poisoned = is_batch_poisoned.to(device)
            else:
                (images, is_batch_poisoned, _, file_index) = content
                is_batch_poisoned = is_batch_poisoned.to(device)

            images = images.to(device)
            if args.siftout_poisoned_images:
                trainset_file_indices.append(file_index)

            if args.num_views == 1:
                views = images.clone()
                views = views.unsqueeze(1)
            else:
                views = generate_view_tensors(images, ss_transform)
            views = views.to(device)

            bs, n_views, c, h, w = views.shape
            views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
            if args.unlearn_before_finding_trigger_channels:
                vision_features = unlearnt_backbone(views)
            else:
                vision_features = backbone(views)  # [bs*n_views, 512]

            if args.normalize_backbone_features == "l2":
                vision_features = F.normalize(vision_features, dim=-1)
            _, C = vision_features.shape

            corrs, max_indices_at_channel = get_ss_statistics(
                vision_features.detach().cpu().numpy(), bs, C, args
            )
            if "frequency_ensemble" in args.bd_detectors:
                get_freq_detection_scores(
                    images, freq_detector_ensemble, bd_detector_scores, args
                )
            if args.detect_projector_features:
                get_detection_scores_from_projector(
                    vision_features, projector, bs, bd_detector_scores, args
                )
            else:
                get_detection_scores(
                    vision_features,
                    corrs,
                    max_indices_at_channel,
                    bd_detector_scores,
                    args,
                )

            all_votes.append(max_indices_at_channel)
            is_poisoned.append(is_batch_poisoned)

    is_poisoned = torch.cat(is_poisoned)
    is_poisoned = np.array(is_poisoned.cpu())  # [#dataset]

    total_images = len(data_loader.dataset)
    minority_lb = int(total_images * args.minority_lower_bound)
    minority_ub = int(total_images * args.minority_upper_bound)

    all_votes = np.concatenate(all_votes, axis=0)  # [#dataset, n_view*take_channel]

    minority_indices = []

    for detector, values in bd_detector_scores.items():
        bd_scores = np.array(values)

        if not args.ideal_case:
            auroc = roc_auc_score(y_true=is_poisoned, y_score=bd_scores)
            print(
                f"the AUROC score of detector '{detector}' is: {np.round(auroc*100,1)}"
            )

        bd_indices = np.argsort(bd_scores)  # indices, sorted from low to high
        if minority_lb > 0:
            minority_indices_local = bd_indices[
                -minority_ub:-minority_lb
            ]  # numpy array
        else:
            minority_indices_local = bd_indices[-minority_ub:]
        minority_indices.extend(minority_indices_local.tolist())

    minority_indices_counter = Counter(minority_indices)
    minority_indices = [
        idx
        for idx, count in minority_indices_counter.items()
        if count in args.in_n_detectors
    ]

    all_votes = all_votes[
        minority_indices
    ]  # votes by minority, [minority_num, n_view*take_channel]

    is_poisoned = is_poisoned[minority_indices]
    poisoned_found = is_poisoned.sum()

    print(
        f"total count of found poisoned images: {poisoned_found}/{is_poisoned.shape[0]}={np.round(poisoned_found/is_poisoned.shape[0]*100,1)}"
    )

    if args.siftout_poisoned_images:
        trainset_file_indices = torch.cat(trainset_file_indices)
        trainset_file_indices = np.array(trainset_file_indices.cpu())  # [#dataset]
        estimated_poisoned_file_indices = trainset_file_indices[minority_indices]
        return estimated_poisoned_file_indices  # numpy

    if args.find_and_ignore_probe_channels:
        # REMOVE channels that appear in probe dataset
        essential_indices = Counter(all_votes.flatten()).most_common(
            max(args.channel_num) + args.ignore_probe_channel_num
        )
        essential_indices = [idx for (idx, occ_count) in essential_indices]

        all_probe_votes = np.concatenate(
            all_probe_votes, axis=0
        )  # [#dataset, n_view*take_channel]
        probe_essential_indices = Counter(all_probe_votes.flatten()).most_common(
            args.ignore_probe_channel_num
        )
        probe_essential_indices = [
            idx for (idx, occ_count) in probe_essential_indices
        ]  # a list of channel indices

        essential_indices = [
            item for item in essential_indices if item not in probe_essential_indices
        ]
        essential_indices = torch.tensor(essential_indices[: max(args.channel_num)])
    else:
        essential_indices = Counter(all_votes.flatten()).most_common(
            max(args.channel_num)
        )
        essential_indices = torch.tensor(
            [idx for (idx, occ_count) in essential_indices]
        )

    if args.full_dataset_svd:
        os.remove(h5py_filename)
    return essential_indices


def get_feats(loader, model, args):

    # switch to evaluate mode
    model.eval()
    feats, ptr = None, 0

    with torch.no_grad():
        for i, content in enumerate(loader):
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
    train_loader,
    backbone,
    linear,
    optimizer,
    args,
):
    backbone.eval()
    linear.train()
    for i, content in enumerate(train_loader):
        (images, target, _) = content

        images = images.to(device)
        target = target.to(device)

        # compute output
        with torch.no_grad():
            output = backbone(images)

        output = linear(output)
        loss = F.cross_entropy(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def produces_evaluation_results(linear, output, target, acc1_accumulator, total_count):
    output = linear(output)
    _, pred = output.topk(
        1, 1, True, True
    )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
    # pred.shape: [bs, k=1]
    pred = pred.squeeze(1)  # shape: [bs, ]

    total_count += target.shape[0]
    acc1_accumulator += (pred == target).float().sum().item()
    return acc1_accumulator, total_count


def eval_linear_classifier(
    val_loader, backbone, linear, args, val_mode, use_ss_detector, contributing_indices
):
    with torch.no_grad():
        if args.detect_trigger_channels and use_ss_detector:
            acc1_accumulator_dict = {}
            total_count_dict = {}
            for k in args.channel_num:
                acc1_accumulator_dict[k] = 0.0
                total_count_dict[k] = 0
        else:
            acc1_accumulator = 0.0
            total_count = 0

        for i, content in enumerate(val_loader):
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
                for k in args.channel_num:
                    indices_toremove = contributing_indices[0:k]
                    output[:, indices_toremove] = 0.0

                    acc1_r, total_r = produces_evaluation_results(
                        linear,
                        output,
                        target,
                        acc1_accumulator_dict[k],
                        total_count_dict[k],
                    )
                    acc1_accumulator_dict[k] = acc1_r
                    total_count_dict[k] = total_r

            else:
                acc1_accumulator, total_count = produces_evaluation_results(
                    linear, output, target, acc1_accumulator, total_count
                )

        if args.detect_trigger_channels and use_ss_detector:
            results_dict = {}
            for k in args.channel_num:
                results_dict[k] = acc1_accumulator_dict[k] / total_count_dict[k] * 100.0
            return results_dict
        else:
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

        # if self.args.detect_trigger_channels:
        #     self.contributing_indices = None

    # Linear Probe training and evalaution
    def linear_probing(
        self,  # call self.args for options
        model,
        poison,
        use_mask_pruning=False,
        trained_linear=None,
        force_training=False,
    ):
        if use_mask_pruning:
            # use mask pruning
            model.eval()

            if self.args.method == "mocov2":
                backbone = copy.deepcopy(model.encoder_q)
                backbone.fc = nn.Sequential()
            else:
                backbone = copy.deepcopy(model.backbone)
            linear = copy.deepcopy(trained_linear)

            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(
                list(backbone.parameters()) + list(linear.parameters()),
                lr=self.args.unlearning_lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args.schedule, gamma=0.1
            )

            #### stage 1: model unlearing
            print(f">>>>>>>> start model unlearning")
            for epoch in range(0, self.args.unlearning_epochs + 1):
                # UNLEARNING
                train_acc = train_step_unlearning(
                    args=self.args,
                    model=backbone,
                    linear=linear,
                    criterion=criterion,
                    optimizer=optimizer,
                    data_loader=poison.train_probe_loader,
                )

                scheduler.step()
                print(f">>>>>>>> at epoch {epoch}, the train_acc is {train_acc}")

                if train_acc <= self.args.clean_threshold:
                    print(
                        f">>>>>>>> arrive at early break of stage 1 unlearning at epoch {epoch}"
                    )
                    # end stage 1
                    break

            #### stage 2: model recovering
            print(f">>>>>>>> start model recovering")
            if self.args.method == "mocov2":
                unlearned_model = models.__dict__[self.args.arch](
                    num_classes=512, norm_layer=MaskBatchNorm2d
                )
                unlearned_model.fc = nn.Sequential()
            else:
                if "cifar" in self.args.dataset or "gtsrb" in self.args.dataset:
                    model_fun, _ = model_dict_cifar[self.args.arch]
                else:
                    model_fun, _ = model_dict[self.args.arch]
                unlearned_model = model_fun(norm_layer=MaskBatchNorm2d)

            refill_unlearned_model(
                unlearned_model, orig_state_dict=backbone.state_dict()
            )

            unlearned_model = unlearned_model.to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)

            parameters = list(unlearned_model.named_parameters())
            mask_params = [
                v for n, v in parameters if "neuron_mask" in n
            ]  # only update neuron_mask ones
            mask_optimizer = torch.optim.SGD(
                mask_params, lr=self.args.recovering_lr, momentum=0.9
            )

            for epoch in range(1, self.args.recovering_epochs + 1):
                train_step_recovering(
                    args=self.args,
                    unlearned_model=unlearned_model,
                    linear=linear,
                    criterion=criterion,
                    data_loader=poison.train_probe_loader,
                    mask_opt=mask_optimizer,
                )

            save_mask_scores(
                unlearned_model.state_dict(),
                os.path.join(self.args.saved_path, "mask_values.txt"),
            )

            del unlearned_model, backbone

            #### stage 3: model pruning
            print(f">>>>>>>> start model pruning")
            # read poisoned model again!
            if self.args.method == "mocov2":
                backbone = copy.deepcopy(model.encoder_q)
                backbone.fc = nn.Sequential()
            else:
                backbone = copy.deepcopy(model.backbone)
            linear = copy.deepcopy(trained_linear)

            criterion = torch.nn.CrossEntropyLoss().to(device)
            mask_file = os.path.join(self.args.saved_path, "mask_values.txt")
            mask_values = read_data(mask_file)
            mask_values = sorted(mask_values, key=lambda x: float(x[2]))
            print("No. \t Layer Name \t Neuron Idx \t Mask \t PoisonACC \t CleanACC")
            cl_loss, cl_acc = test_maskprune(
                args=self.args,
                model=backbone,
                linear=linear,
                criterion=criterion,
                data_loader=poison.test_clean_loader,
                val_mode="clean",
            )
            po_loss, po_acc = test_maskprune(
                args=self.args,
                model=backbone,
                linear=linear,
                criterion=criterion,
                data_loader=poison.test_pos_loader,
                val_mode="poison",
            )
            print(
                "0 \t None     \t None  \t None   \t {:.4f} \t {:.4f}".format(
                    # po_loss,
                    po_acc * 100,
                    # cl_loss,
                    cl_acc * 100,
                )
            )  # this records the backdoored model's initial results

            if self.args.pruning_by == "threshold":
                evaluate_by_threshold(
                    self.args,
                    backbone,
                    linear,
                    mask_values,
                    pruning_max=self.args.pruning_max,
                    pruning_step=self.args.pruning_step,
                    criterion=criterion,
                    clean_loader=poison.test_clean_loader,
                    poison_loader=poison.test_pos_loader,
                )
            else:
                raise Exception("Not implemented yet")

        else:
            # NOT USING MASK PRUNING
            model.eval()
            if self.args.method == "mocov2":
                backbone = copy.deepcopy(model.encoder_q)
                backbone.fc = nn.Sequential()
            else:
                backbone = model.backbone

            if "cifar" in self.args.dataset or "gtsrb" in self.args.dataset:
                _, feat_dim = model_dict_cifar[self.args.arch]
            else:
                _, feat_dim = model_dict[self.args.arch]

            # train_probe_feats_mean = None
            if (
                self.args.linear_probe_normalize == "ref_set"
                or self.args.replacement_value == "ref_mean"
            ):
                train_probe_feats = get_feats(
                    poison.train_probe_loader, backbone, self.args
                )  # shape: ? [N, D]
                # train_probe_feats_mean = torch.mean(
                #     train_probe_feats, dim=0
                # )  # shape: [D, ], used if replacement_value == "ref_mean"

            # training linear
            if self.args.linear_probe_normalize == "ref_set":
                train_var, train_mean = torch.var_mean(train_probe_feats, dim=0)

                linear = nn.Sequential(
                    Normalize(),  # L2 norm
                    FullBatchNorm(
                        train_var, train_mean
                    ),  # the train_var/mean are from L2-normed features
                    nn.Linear(feat_dim, self.args.num_classes),
                )
            elif self.args.linear_probe_normalize == "batch":
                linear = nn.Sequential(
                    nn.BatchNorm1d(feat_dim, affine=False),
                    nn.Linear(feat_dim, self.args.num_classes),
                )
            elif self.args.linear_probe_normalize == "none":
                linear = nn.Linear(
                    feat_dim, self.args.num_classes
                )  # FIXME: tune learning rate
            elif self.args.linear_probe_normalize == "regular":
                linear = nn.Sequential(
                    Normalize(),  # L2 norm
                    nn.Linear(feat_dim, self.args.num_classes),
                )

            if self.args.pretrained_linear_model != "":
                pretrained_state_dict = torch.load(
                    self.args.pretrained_linear_model, map_location=device
                )
                linear.load_state_dict(pretrained_state_dict, strict=True)

            linear = linear.to(device)

            if self.args.pretrained_linear_model == "" or force_training:
                optimizer = torch.optim.SGD(
                    linear.parameters(),
                    lr=0.06,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
                sched = [15, 30, 40]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=sched
                )

                # train linear classifier
                linear_probing_epochs = 40

                for epoch in range(linear_probing_epochs):
                    print(f"training linear classifier, epoch: {epoch}")
                    train_linear_classifier(
                        poison.train_probe_loader,
                        backbone,
                        linear,
                        optimizer,
                        self.args,
                    )
                    # modify lr
                    lr_scheduler.step()

                if not self.args.distributed or (
                    self.args.distributed
                    and self.args.local_rank % self.args.ngpus_per_node == 0
                ):
                    save_model(
                        linear.state_dict(),
                        filename=os.path.join(self.args.saved_path, "linear.pth.tar"),
                    )

            backbone.eval()
            linear.eval()

            print(f"<<<<<<<<< evaluating linear on CLEAN val")
            clean_acc1 = eval_linear_classifier(
                poison.test_clean_loader,
                backbone,
                linear,
                self.args,
                val_mode="clean",
                use_ss_detector=False,
                contributing_indices=None,
                # contributing_indices=self.contributing_indices,
            )

            print(f"<<<<<<<<< evaluating linear on POISON val")
            poison_acc1 = eval_linear_classifier(
                poison.test_pos_loader,
                backbone,
                linear,
                self.args,
                val_mode="poison",
                use_ss_detector=False,
                contributing_indices=None,
                # contributing_indices=self.contributing_indices,
            )

            print(
                f"with the DEFAULT linear classifier, the ACC on clean val is: {np.round(clean_acc1,1)}, the ASR on poisoned val is: {np.round(poison_acc1,1)}"
            )

            return linear  # the returned linear is only used if use_ss_detector=False

    # SSL attack and kNN Evaluation
    def train_freq(
        self, model, optimizer, train_transform, poison, force_training=False
    ):

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
        test_clean_loader = poison.test_clean_loader  # clean val
        test_back_loader = poison.test_pos_loader  # poisoned val (test) set

        clean_acc = 0.0
        back_acc = 0.0

        for epoch in range(self.args.start_epoch, self.args.epochs):
            losses = AverageMeter()
            cl_losses = AverageMeter()

            train_transform = train_transform.to(device)

            # 1 epoch training
            start = time.time()

            # TRAIN
            if self.args.pretrained_ssl_model == "" or force_training:
                for i, content in enumerate(
                    train_loader
                ):  # frequency backdoor has been injected
                    if self.args.detect_trigger_channels:
                        (images, is_poisoned, __, _) = content
                    else:
                        (images, __, _) = content
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

                    losses.update(loss.item(), images[0].size(0))
                    cl_losses.update(loss.item(), images[0].size(0))

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                warmup_scheduler.step()

            # EVAL
            if epoch + 1 == self.args.epochs or (
                (self.args.pretrained_ssl_model == "" or force_training)
                and epoch % self.args.knn_eval_freq == 0
            ):
                model.eval()

                if self.args.method == "mocov2":
                    backbone = copy.deepcopy(model.encoder_q)
                    backbone.fc = nn.Sequential()
                else:
                    backbone = model.backbone

                clean_acc, back_acc = self.knn_monitor_fre(
                    backbone,
                    poison.memory_loader,
                    test_clean_loader,
                    self.args,
                    classes=self.args.num_classes,
                    subset=False,
                    backdoor_loader=test_back_loader,
                )
                print(
                    "[{}-epoch] time:{:.1f} | clean acc: {:.1f} | back acc: {:.1f} | loss:{:.3f} | cl_loss:{:.3f}".format(
                        epoch + 1,
                        time.time() - start,
                        clean_acc,
                        back_acc,
                        losses.avg,
                        cl_losses.avg,
                    )
                )

        if self.args.pretrained_ssl_model == "" or force_training:
            # Save final model
            if not self.args.distributed or (
                self.args.distributed
                and self.args.local_rank % self.args.ngpus_per_node == 0
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

    # sift out poisoned images
    def siftout_poisoned_images(self, model, poison, trained_linear):

        linear = copy.deepcopy(trained_linear)
        linear.eval()

        model.eval()
        if self.args.method == "mocov2":
            backbone = copy.deepcopy(model.encoder_q)
            backbone.fc = nn.Sequential()
            # FIXME: projector
        else:
            backbone = model.backbone
            projector = model.proj_head  # FIXME: BYOL may use different name

        backbone.eval()
        projector.eval()

        estimated_poisoned_file_indices = find_trigger_channels(
            self.args,
            poison.train_pos_loader,
            poison.train_probe_loader,
            poison.train_probe_freq_detector_loader,
            backbone,
            projector,
            linear,
            poison.ss_transform,
        )  # numpy
        return estimated_poisoned_file_indices

    # Channel Voting Strategy
    def trigger_channel_removal(self, model, poison, trained_linear):
        ######## Prepare backbone and linear, and set them to eval mode
        linear = copy.deepcopy(trained_linear)
        linear.eval()

        model.eval()
        if self.args.method == "mocov2":
            backbone = copy.deepcopy(model.encoder_q)
            backbone.fc = nn.Sequential()
            # FIXME: projector
        else:
            backbone = model.backbone
            projector = model.proj_head  # FIXME: BYOL may use different name

        backbone.eval()
        projector.eval()

        if self.args.ideal_case:
            clean_val_contributing_indices = find_trigger_channels(
                self.args,
                poison.test_clean_loader,  # poisoned training set
                poison.train_probe_loader,  # 1% clean train probe dataset
                poison.train_probe_freq_detector_loader,  # same to train_probe_loader, only batch size is fxied to 64
                backbone,
                projector,
                linear,
                poison.ss_transform,
            )
            poi_val_contributing_indices = find_trigger_channels(
                self.args,
                poison.test_pos_loader,  # poisoned training set
                poison.train_probe_loader,  # 1% clean train probe dataset
                poison.train_probe_freq_detector_loader,  # same to train_probe_loader, only batch size is fxied to 64
                backbone,
                projector,
                linear,
                poison.ss_transform,
            )
            ############# KNN
            clean_acc_SSDETECTOR, back_acc_SSDETECTOR = self.knn_monitor_fre(
                backbone,
                poison.memory_loader,
                poison.test_clean_loader,
                self.args,
                classes=self.args.num_classes,
                subset=False,
                backdoor_loader=poison.test_pos_loader,
                use_SS_detector=True,
                clean_val_contributing_indices=clean_val_contributing_indices,
                poi_val_contributing_indices=poi_val_contributing_indices,
            )

            for k in self.args.channel_num:
                print(
                    f"In kNN classification, by replacing top-{k} channels, clean acc: {clean_acc_SSDETECTOR[k]:.1f} | back acc: {back_acc_SSDETECTOR[k]:.1f}"
                )

            ########### Linear Probe
            print(f"<<<<<<<<< evaluating linear on CLEAN val")
            clean_acc1 = eval_linear_classifier(
                poison.test_clean_loader,
                backbone,
                linear,
                self.args,
                val_mode="clean",
                use_ss_detector=True,
                contributing_indices=clean_val_contributing_indices,
            )

            print(f"<<<<<<<<< evaluating linear on POISON val")
            poison_acc1 = eval_linear_classifier(
                poison.test_pos_loader,
                backbone,
                linear,
                self.args,
                val_mode="poison",
                use_ss_detector=True,
                contributing_indices=poi_val_contributing_indices,
            )
            for k in self.args.channel_num:
                print(
                    f"In linear probe, by replacing {k} channels, the ACC on clean val is: {np.round(clean_acc1[k],1)}, the ASR on poisoned val is: {np.round(poison_acc1[k],1)}"
                )

        else:
            ######## Find trigger channels in REALISTIC case (i.e., find channel from poisoned train set)
            contributing_indices = find_trigger_channels(
                self.args,
                poison.train_pos_loader,  # poisoned training set
                poison.train_probe_loader,  # 1% clean train probe dataset
                poison.train_probe_freq_detector_loader,  # same to train_probe_loader, only batch size is fxied to 64
                backbone,
                projector,
                linear,
                poison.ss_transform,
            )

            ############# KNN
            clean_acc_SSDETECTOR, back_acc_SSDETECTOR = self.knn_monitor_fre(
                backbone,
                poison.memory_loader,
                poison.test_clean_loader,
                self.args,
                classes=self.args.num_classes,
                subset=False,
                backdoor_loader=poison.test_pos_loader,
                use_SS_detector=True,
                contributing_indices=contributing_indices,
            )

            for k in self.args.channel_num:
                print(
                    f"In kNN classification, by replacing top-{k} channels, clean acc: {clean_acc_SSDETECTOR[k]:.1f} | back acc: {back_acc_SSDETECTOR[k]:.1f}"
                )

            ########### Linear Probe
            print(f"<<<<<<<<< evaluating linear on CLEAN val")
            clean_acc1 = eval_linear_classifier(
                poison.test_clean_loader,
                backbone,
                linear,
                self.args,
                val_mode="clean",
                use_ss_detector=True,
                contributing_indices=contributing_indices,
            )

            print(f"<<<<<<<<< evaluating linear on POISON val")
            poison_acc1 = eval_linear_classifier(
                poison.test_pos_loader,
                backbone,
                linear,
                self.args,
                val_mode="poison",
                use_ss_detector=True,
                contributing_indices=contributing_indices,
            )
            for k in self.args.channel_num:
                print(
                    f"In linear probe, by replacing {k} channels, the ACC on clean val is: {np.round(clean_acc1[k],1)}, the ASR on poisoned val is: {np.round(poison_acc1[k],1)}"
                )

    @torch.no_grad()
    def knn_monitor_fre(
        self,
        net,
        memory_data_loader,
        test_data_loader,
        args,
        k=200,
        t=0.1,
        hide_progress=True,
        classes=-1,
        subset=False,
        backdoor_loader=None,
        use_SS_detector=False,
        contributing_indices=None,
        clean_val_contributing_indices=None,
        poi_val_contributing_indices=None,
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

        if args.detect_trigger_channels and args.replacement_value == "ref_mean":
            feature_bank_mean = torch.mean(feature_bank, dim=1)  # shape: [D, ]

        # feature_labels: [total num]
        feature_labels = (
            memory_data_loader.dataset[:][1].clone().detach().to(feature_bank.device)
        )

        """
        Evaluate clean KNN
        """
        print(">>>>>>> now KNN evaluate for CLEAN val")
        if use_SS_detector:
            clean_val_top1_dict = {}
            clean_val_total_num_dict = {}
            for k in args.channel_num:
                clean_val_top1_dict[k] = 0.0
                clean_val_total_num_dict[k] = 0
        else:
            clean_val_top1, clean_val_total_num = 0.0, 0

        test_bar = tqdm(test_data_loader, desc="kNN", disable=hide_progress)
        for content in test_bar:

            (data, target, _) = content

            data, target = data.to(device), target.to(device)
            feature = net(data)

            if use_SS_detector:
                for k in args.channel_num:

                    indices_toremove = (
                        clean_val_contributing_indices[0:k]
                        if args.ideal_case
                        else contributing_indices[0:k]
                    )
                    feature[:, indices_toremove] = 0.0
                    feature = F.normalize(feature, dim=1)
                    pred_labels = self.knn_predict(
                        feature, feature_bank, feature_labels, classes, k, t
                    )
                    clean_val_total_num_dict[k] = clean_val_total_num_dict[
                        k
                    ] + data.size(0)
                    clean_val_top1_dict[k] = (
                        clean_val_top1_dict[k]
                        + (pred_labels[:, 0] == target).float().sum().item()
                    )
            else:
                feature = F.normalize(feature, dim=1)
                # feature: [bsz, dim]
                pred_labels = self.knn_predict(
                    feature, feature_bank, feature_labels, classes, k, t
                )

                clean_val_total_num += data.size(0)
                clean_val_top1 += (pred_labels[:, 0] == target).float().sum().item()
                # test_bar.set_postfix(
                #     {"Accuracy": clean_val_top1 / clean_val_total_num * 100}
                # )

        """
        Evaluate poison KNN
        """
        print(">>>>>>> now KNN evaluate for POISON val")
        if use_SS_detector:
            backdoor_val_top1_dict = {}
            backdoor_val_total_num_dict = {}
            for k in args.channel_num:
                backdoor_val_top1_dict[k] = 0.0
                backdoor_val_total_num_dict[k] = 0
        else:
            backdoor_val_top1, backdoor_val_total_num = 0.0, 0

        backdoor_test_bar = tqdm(backdoor_loader, desc="kNN", disable=hide_progress)

        for content in backdoor_test_bar:
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
                for k in args.channel_num:
                    indices_toremove = (
                        poi_val_contributing_indices[0:k]
                        if args.ideal_case
                        else contributing_indices[0:k]
                    )
                    feature[:, indices_toremove] = 0.0
                    feature = F.normalize(feature, dim=1)
                    pred_labels = self.knn_predict(
                        feature, feature_bank, feature_labels, classes, k, t
                    )
                    backdoor_val_total_num_dict[k] = backdoor_val_total_num_dict[
                        k
                    ] + data.size(0)
                    backdoor_val_top1_dict[k] = (
                        backdoor_val_top1_dict[k]
                        + (pred_labels[:, 0] == target).float().sum().item()
                    )
            else:
                feature = F.normalize(feature, dim=1)
                # feature: [bsz, dim]
                pred_labels = self.knn_predict(
                    feature, feature_bank, feature_labels, classes, k, t
                )

                backdoor_val_total_num += data.size(0)
                backdoor_val_top1 += (pred_labels[:, 0] == target).float().sum().item()
                # test_bar.set_postfix(
                #     {"Accuracy": backdoor_val_top1 / backdoor_val_total_num * 100}
                # )
        if use_SS_detector:
            clean_results_dict = {}
            backdoor_results_dict = {}
            for k in args.channel_num:
                clean_results_dict[k] = (
                    clean_val_top1_dict[k] / clean_val_total_num_dict[k] * 100.0
                )
                backdoor_results_dict[k] = (
                    backdoor_val_top1_dict[k] / backdoor_val_total_num_dict[k] * 100.0
                )
            return clean_results_dict, backdoor_results_dict
        else:

            return (
                clean_val_top1 / clean_val_total_num * 100,
                backdoor_val_top1 / backdoor_val_total_num * 100,
            )

    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # feature: [bsz, dim]
        # feature_bank: [dim, clean_val_total_num]
        # feature_labels: [clean_val_total_num]

        # #  REMOVE LATER
        # print(f"[___DEBUG___]: ==============================")
        # print(f"[___DEBUG___]: feature.shape: {feature.shape}")
        # print(f"[___DEBUG___]: feature_bank.shape: {feature_bank.shape}")
        # print(f"[___DEBUG___]: feature_labels.shape: {feature_labels.shape}")

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
        # print(f"[___DEBUG___]: one_hot_label.shape: {one_hot_label.shape}")
        # one_hot_label: [bsz*K, C]
        one_hot_label = one_hot_label.scatter(
            dim=-1, index=sim_labels.view(-1, 1), value=1.0
        )  # for each row, only one column is 1, which is the label of k-nearest this neighbor
        # print(f"[___DEBUG___]: one_hot_label.shape: {one_hot_label.shape}")

        # weighted score ---> [bsz, C]
        pred_scores = torch.sum(
            one_hot_label.view(feature.size(0), -1, classes)  # [bs, k, C=Classes]
            * sim_weight.unsqueeze(dim=-1),  # [bs, k, 1]
            dim=1,
        )  # [bs, C], where each column means the SCORE (weight) of the sample to the class at this column index

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels  # [bs, C], where the first column is the index (class) of nearest cluster
