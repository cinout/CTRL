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
from collections import Counter, OrderedDict
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar
from utils.util import AverageMeter, save_model
from utils.knn import knn_monitor
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
from networks.mask_batchnorm import MaskBatchNorm2d
import pandas as pd
import PIL
import random
from frequency_detector import FrequencyDetector, patching_train, dct2


device = "cuda" if torch.cuda.is_available() else "cpu"


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = "{}.{}".format(neuron[0], "weight")
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


# for evaluating performances at different stages
def test_maskprune(args, model, linear, criterion, data_loader, val_mode):
    model.eval()
    linear.eval()

    total_correct = 0
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for content in data_loader:
            # if args.detect_trigger_channels:
            #     if val_mode == "poison":
            #         (images, views, labels, original_label, _) = content
            #         original_label = original_label.to(device)
            #     elif val_mode == "clean":
            #         (images, views, labels, _) = content
            #     else:
            #         raise Exception(f"unimplemented val_mode {val_mode}")
            # else:
            if val_mode == "poison":
                (images, labels, original_label, _) = content
                original_label = original_label.to(device)
            elif val_mode == "clean":
                (images, labels, _) = content
            else:
                raise Exception(f"unimplemented val_mode {val_mode}")

            images, labels = images.to(device), labels.to(device)
            if val_mode == "poison":
                valid_indices = original_label != args.target_class
                if torch.all(~valid_indices):
                    # all inputs are from target class, skip this iteration
                    continue

                images = images[valid_indices]
                labels = labels[valid_indices]

            output = model(images)
            output = linear(output)

            total_loss += criterion(output, labels).item()

            _, pred = output.topk(
                1, 1, True, True
            )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
            # pred.shape: [bs, k=1]
            pred = pred.squeeze(1)  # shape: [bs, ]
            total_count += labels.shape[0]
            total_correct += (pred == labels).float().sum().item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / total_count
    return loss, acc


# called at 3rd pruning stage
def evaluate_by_threshold(
    args,
    model,
    linear,
    mask_values,  # sorted by [2], from low to high
    pruning_max,  # 0.9
    pruning_step,  # 0.05
    criterion,
    clean_loader,
    poison_loader,
):
    model.eval()
    linear.eval()

    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0  # prune from which idx in mask_values
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = (
            mask_values[idx][0],
            mask_values[idx][1],
            mask_values[idx][2],
        )
        cl_loss, cl_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=clean_loader,
            val_mode="clean",
        )
        po_loss, po_acc = test_maskprune(
            args=args,
            model=model,
            linear=linear,
            criterion=criterion,
            data_loader=poison_loader,
            val_mode="poison",
        )
        print(
            "{} \t {} \t {} \t {:.2f} \t {:.4f} \t {:.4f}".format(
                start,
                layer_name,
                neuron_idx,
                threshold,
                # po_loss,
                po_acc * 100,
                # cl_loss,
                cl_acc * 100,
            )
        )


# called at 3rd stage to read mask (use mask_values.txt as reference)
def read_data(file_name):
    tempt = pd.read_csv(file_name, sep="\s+", skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


# called at the end of 2nd stage
def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if "neuron_mask" in name:
            for idx in range(param.size(0)):
                neuron_name = ".".join(name.split(".")[:-1])
                mask_values.append(
                    "{} \t {} \t {} \t {:.4f} \n".format(
                        count, neuron_name, idx, param[idx].item()
                    )
                )
                count += 1
    with open(file_name, "w") as f:
        f.write("No \t Layer Name \t Neuron Idx \t Mask Score \n")
        f.writelines(mask_values)


# clip value to be witihin 0 and 1
def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [
        param
        for name, param in unlearned_model.named_parameters()
        if "neuron_mask" in name
    ]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def refill_unlearned_model(net, orig_state_dict):
    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            # print(f">>>>>> IN orig_state_dict: {k}")
            new_state_dict[k] = orig_state_dict[k]
        else:
            # print(f">>>>>> OUT orig_state_dict: {k}")
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def train_step_recovering(
    args, unlearned_model, linear, criterion, mask_opt, data_loader
):
    unlearned_model.train()
    linear.train()

    for content in data_loader:

        images, labels, _ = content

        images, labels = images.to(device), labels.to(device)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        output = linear(output)

        loss = criterion(output, labels)
        loss = args.alpha * loss

        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)


def train_step_unlearning(args, model, linear, criterion, optimizer, data_loader):
    model.train()
    linear.train()
    total_correct = 0
    total_count = 0
    for content in data_loader:
        images, labels, _ = content

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        output = linear(output)

        loss = criterion(output, labels)

        _, pred = output.topk(
            1, 1, True, True
        )  # k=1, dim=1, largest, sorted; pred is the indices of largest class
        # pred.shape: [bs, k=1]
        pred = pred.squeeze(1)  # shape: [bs, ]

        total_correct += (pred == labels).float().sum().item()
        total_count += labels.shape[0]

        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(linear.parameters()),
            max_norm=20,
            norm_type=2,
        )
        (-loss).backward()
        optimizer.step()

    acc = float(total_correct) / total_count
    return acc


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
    ss_transform,
):
    all_entropies = []  # for all images in the dataset
    all_votes = []  # for all images in the dataset
    is_poisoned = []  # for all images in the dataset

    total_images = 0

    if args.use_frequency_detector:
        all_frequencies = []  # for all images in the dataset
        freq_detector = FrequencyDetector(height=args.image_size, width=args.image_size)
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
                        images_poi[i] = patching_train(images_clean[i], images_clean)

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
                filename=os.path.join(args.saved_path, "freq_detector.pth.tar"),
            )
        else:
            # load model
            pretrained_state_dict = torch.load(
                args.pretrained_frequency_model, map_location=device
            )
            freq_detector.load_state_dict(pretrained_state_dict, strict=True)

    all_probe_votes = []
    for i, content in enumerate(train_probe_loader):
        (images, target, _) = content

        images = images.to(device)
        views = generate_view_tensors(images, ss_transform)

        views = views.to(device)

        bs, n_views, c, h, w = views.shape
        views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
        vision_features = backbone(views)  # [bs*n_views, 512]
        _, C = vision_features.shape
        vision_features = vision_features.detach().cpu().numpy()
        u, s, v = np.linalg.svd(
            vision_features - np.mean(vision_features, axis=0, keepdims=True),
            full_matrices=False,
        )

        # get top eigenvector
        eig_for_indexing = v[0:1]  # [1, C]

        # adjust direction (sign)
        corrs = np.matmul(
            eig_for_indexing, np.transpose(vision_features)
        )  # [1, bs*n_view]
        coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
        coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
        elementwise = (
            eig_for_indexing * vision_features * coeff_adjust
        )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign

        # get contributing indices sorted from low to high
        max_indices = np.argsort(
            elementwise, axis=1
        )  # [bs*n_view, C], C are indices, sorted by value from low to high
        # total_images += bs
        max_indices = max_indices.reshape(bs, n_views, C)  # [bs, n_view, C]

        max_indices_at_channel = max_indices[
            :, :, -max(args.channel_num) :
        ]  # [bs, n_view, channel_num]
        max_indices_at_channel = max_indices_at_channel.reshape(
            bs, -1
        )  # [bs, n_view*channel_num]
        all_probe_votes.append(max_indices_at_channel)

    for i, content in tqdm(enumerate(data_loader)):
        (images, is_batch_poisoned, _, _) = content
        is_batch_poisoned = is_batch_poisoned.to(device)

        images = images.to(device)

        views = generate_view_tensors(images, ss_transform)
        views = views.to(device)

        bs, n_views, c, h, w = views.shape
        views = views.reshape(-1, c, h, w)  # [bs*n_views, c, h, w]
        vision_features = backbone(views)  # [bs*n_views, 512]
        _, C = vision_features.shape
        vision_features = vision_features.detach().cpu().numpy()
        u, s, v = np.linalg.svd(
            vision_features - np.mean(vision_features, axis=0, keepdims=True),
            full_matrices=False,
        )

        # get top eigenvector
        eig_for_indexing = v[0:1]  # [1, C]

        # adjust direction (sign)
        corrs = np.matmul(
            eig_for_indexing, np.transpose(vision_features)
        )  # [1, bs*n_view]
        coeff_adjust = np.where(corrs > 0, 1, -1)  # [1, bs*n_view]
        coeff_adjust = np.transpose(coeff_adjust)  # [bs*n_view, 1]
        elementwise = (
            eig_for_indexing * vision_features * coeff_adjust
        )  # [bs*n_view, C]; if corrs is negative, then adjust its elements to reverse sign

        # get contributing indices sorted from low to high
        max_indices = np.argsort(
            elementwise, axis=1
        )  # [bs*n_view, C], C are indices, sorted by value from low to high
        total_images += bs
        max_indices = max_indices.reshape(bs, n_views, C)  # [bs, n_view, C]

        max_indices_at_channel = max_indices[
            :, :, -max(args.channel_num) :
        ]  # [bs, n_view, channel_num]
        max_indices_at_channel = max_indices_at_channel.reshape(
            bs, -1
        )  # [bs, n_view*channel_num]

        entropies = []  # bs elements
        if args.minority_criterion == "entropy":
            for votes in max_indices_at_channel:  # for each original image
                votes_counter = Counter(votes).most_common()
                counts = np.array([c for (name, c) in votes_counter])
                p = counts / counts.sum()
                h = -np.sum(p * np.log(p))
                entropy = np.exp(h)
                entropies.append(entropy)
        elif args.minority_criterion == "ss_score":
            corrs = np.abs(corrs)
            corrs = corrs.reshape(-1, n_views)  #  [bs,n_views]
            ss_scores = -1 * np.max(corrs, axis=1)  # [bs]
            entropies.extend(ss_scores.tolist())
        elif args.minority_criterion == "ss_score_elements":
            num_interested_channels = 1  # FIXME:  changeale
            top_channel_votes = max_indices[
                :, :, -num_interested_channels:
            ].flatten()  # [bs*n_view*num_interested_channels]
            votes_of_batch = Counter(top_channel_votes).most_common(
                num_interested_channels
            )
            chosen_channels = [idx for (idx, occ_count) in votes_of_batch]
            scores = elementwise[:, chosen_channels]
            scores = np.sum(scores, axis=1)  # [bs*n_view, ]

            scores = scores.reshape(-1, n_views)  # [ bs, n_views]
            ss_scores = -1 * np.max(scores, axis=1)  # [bs]
            entropies.extend(ss_scores.tolist())

        all_entropies.extend(entropies)
        all_votes.append(max_indices_at_channel)
        is_poisoned.append(is_batch_poisoned)

        if args.use_frequency_detector:
            # evaluate
            freq_detector.eval()
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
            output = freq_detector(
                images
            )  # [bs, 2], the second element is anomaly score
            output = output[:, 1].detach().cpu().tolist()
            all_frequencies.extend(output)

    all_entropies = np.array(all_entropies)
    is_poisoned = torch.cat(is_poisoned)
    is_poisoned = np.array(is_poisoned.cpu())  # [#dataset]
    score = roc_auc_score(y_true=is_poisoned, y_score=-all_entropies)
    print(f"the AUROC score of all_entropies is: {score*100}")

    if args.use_frequency_detector:
        all_frequencies = np.array(all_frequencies)
        freq_auc_score = roc_auc_score(y_true=is_poisoned, y_score=all_frequencies)
        print(f"the AUROC score of frequency detector is: {freq_auc_score*100}")

        # TODO: remove later
        exit()

    all_entropies_indices = np.argsort(
        all_entropies
    )  # indices, sorted from low to high by entropy value

    # minority_num = int(total_images * args.minority_percent)
    minority_lb = int(total_images * args.minority_percent_lower_bound)
    minority_ub = int(total_images * args.minority_percent_upper_bound)
    minority_num = minority_ub - minority_lb

    # minority_indices = all_entropies_indices[:minority_num]
    minority_indices = all_entropies_indices[minority_lb:minority_ub]

    all_votes = np.concatenate(all_votes, axis=0)  # [#dataset, n_view*channel_num]

    all_votes = all_votes[
        minority_indices
    ]  # votes by minority, [minority_num, n_view*channel_num]

    is_poisoned = is_poisoned[minority_indices]
    poisoned_found = is_poisoned.sum()
    print(
        f"total count of found poisoned images: {poisoned_found}/{is_poisoned.shape[0]}={np.round(poisoned_found/is_poisoned.shape[0]*100,2)}"
    )

    # obtain trigger channels
    essential_indices = Counter(all_votes.flatten()).most_common(
        2 * max(args.channel_num)
    )

    print(
        f"essential_indices: {essential_indices}; #samples: {minority_num*args.num_views*max(args.channel_num)}"
    )

    print(
        f"lowest entropies are: {[ round(item,2) for item in all_entropies[minority_indices]]}"
    )
    print(
        f"entropy mean is {np.mean(all_entropies):.2f}, std is {np.std(all_entropies):.2f}"
    )
    essential_indices = [idx for (idx, occ_count) in essential_indices]

    # FIXME: remove all_probe_votes from all_votes
    all_probe_votes = np.concatenate(
        all_probe_votes, axis=0
    )  # [#dataset, n_view*channel_num]
    probe_essential_indices = Counter(all_probe_votes.flatten()).most_common(
        max(args.channel_num)
    )
    probe_essential_indices = [
        idx for (idx, occ_count) in probe_essential_indices
    ]  # a list of channel indices

    print(f"probe_essential_indices are: {probe_essential_indices}")

    essential_indices = [
        item for item in essential_indices if item not in probe_essential_indices
    ]

    essential_indices = torch.tensor(essential_indices[: max(args.channel_num)])

    print(f"after removing probe channels, essential_indices are: {essential_indices}")

    # FIXME: end of removing

    return essential_indices


# DISABLED, because it makes 0-channel_mean not 0, which is not good for our SS detecting strategy
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
            # if args.detect_trigger_channels:
            #     if val_mode == "poison":
            #         (images, views, target, original_label, _) = content
            #         original_label = original_label.to(device)
            #     elif val_mode == "clean":
            #         (images, views, target, _) = content
            #     else:
            #         raise Exception(f"unimplemented val_mode {val_mode}")
            # else:
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

        if self.args.detect_trigger_channels:
            self.contributing_indices = None

    def linear_probing(
        self,  # call self.args for options
        model,
        poison,
        use_ss_detector=False,
        use_mask_pruning=False,
        trained_linear=None,
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
                data_loader=poison.test_loader,
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
                    clean_loader=poison.test_loader,
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

            train_probe_feats_mean = None
            if self.args.use_ref_norm or self.args.replacement_value == "ref_mean":
                train_probe_feats = get_feats(
                    poison.train_probe_loader, backbone, self.args
                )  # shape: ? [N, D]
                train_probe_feats_mean = torch.mean(
                    train_probe_feats, dim=0
                )  # shape: [D, ], used if replacement_value == "ref_mean"

            if use_ss_detector:
                # it means a linear classifier is already trained in the last step
                linear = copy.deepcopy(trained_linear)
            else:
                # training linear

                if self.args.use_ref_norm:
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

                if self.args.pretrained_linear_model != "":
                    pretrained_state_dict = torch.load(
                        self.args.pretrained_linear_model, map_location=device
                    )
                    linear.load_state_dict(pretrained_state_dict, strict=True)

                linear = linear.to(device)

                if self.args.pretrained_linear_model == "":
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
                            filename=os.path.join(
                                self.args.saved_path, "linear.pth.tar"
                            ),
                        )

            # eval linear classifier
            backbone.eval()
            linear.eval()

            print(f"<<<<<<<<< evaluating linear on CLEAN val")
            clean_acc1 = eval_linear_classifier(
                poison.test_loader,
                backbone,
                linear,
                self.args,
                val_mode="clean",
                use_ss_detector=use_ss_detector,
                contributing_indices=self.contributing_indices,
            )

            print(f"<<<<<<<<< evaluating linear on POISON val")
            poison_acc1 = eval_linear_classifier(
                poison.test_pos_loader,
                backbone,
                linear,
                self.args,
                val_mode="poison",
                use_ss_detector=use_ss_detector,
                contributing_indices=self.contributing_indices,
            )

            if use_ss_detector:
                for k in self.args.channel_num:
                    print(
                        f"by replacing {k} channels, the ACC on clean val is: {clean_acc1[k]}, the ASR on poisoned val is: {poison_acc1[k]}"
                    )
            else:

                print(
                    f"with the DEFAULT linear classifier, the ACC on clean val is: {clean_acc1}, the ASR on poisoned val is: {poison_acc1}"
                )

            return linear  # the returned linear is only used if use_ss_detector=False

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
            losses = AverageMeter()
            cl_losses = AverageMeter()

            train_transform = train_transform.to(device)

            # 1 epoch training
            start = time.time()

            # this is where training occurs
            if self.args.pretrained_ssl_model == "":
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

            # (KNN-eval) why this eval step? (this code combines training and eval together)
            if epoch + 1 == self.args.epochs or (
                self.args.pretrained_ssl_model == ""
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

                # Apply channel removal (our method) to see its efficacy in KNN classification
                if epoch + 1 == self.args.epochs and self.args.detect_trigger_channels:
                    # if last epoch, also evaluate with SS detctor

                    model.eval()
                    if self.args.method == "mocov2":
                        backbone = copy.deepcopy(model.encoder_q)
                        backbone.fc = nn.Sequential()
                    else:
                        backbone = model.backbone

                    self.contributing_indices = find_trigger_channels(
                        self.args,
                        poison.train_pos_loader,
                        poison.train_probe_loader,
                        poison.train_probe_freq_detector_loader,
                        backbone,
                        poison.ss_transform,
                    )

                    clean_acc_SSDETECTOR, back_acc_SSDETECTOR = self.knn_monitor_fre(
                        backbone,
                        poison.memory_loader,
                        test_loader,
                        epoch,
                        self.args,
                        classes=self.args.num_classes,
                        subset=False,
                        backdoor_loader=test_back_loader,
                        use_SS_detector=True,
                        contributing_indices=self.contributing_indices,
                    )

                    for k in self.args.channel_num:
                        print(
                            f"In kNN classification, by replacing top-{k} channels, clean acc: {clean_acc_SSDETECTOR[k]:.3f} | back acc: {back_acc_SSDETECTOR[k]:.3f}"
                        )

        if self.args.pretrained_ssl_model == "":
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
        contributing_indices=None,
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
                    indices_toremove = contributing_indices[0:k]
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
                    indices_toremove = contributing_indices[0:k]
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
