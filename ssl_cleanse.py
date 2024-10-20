import torch.nn.functional as F
from torch.utils import data
import torch
import torchvision
import copy
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
import os
from torch.optim.lr_scheduler import MultiStepLR

device = "cuda" if torch.cuda.is_available() else "cpu"


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


class DatasetCluster(data.Dataset):
    def __init__(self, rep_target, x_other_sample):
        self.rep_target = rep_target
        self.x_other_sample = x_other_sample
        self.rep_target_indices = torch.randint(
            0, rep_target.shape[0], (x_other_sample.shape[0],)
        )

    def __getitem__(self, idx):
        image = self.x_other_sample[idx]
        rep_target = self.rep_target[self.rep_target_indices[idx]]

        return image, rep_target

    def __len__(self):
        return self.x_other_sample.shape[0]


def dataloader_cluster(args, rep_target, x_other_sample, batch_size):
    return data.DataLoader(
        dataset=DatasetCluster(rep_target, x_other_sample),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def draw(base, mean, std, mask, delta):
    delta_norm = torchvision.transforms.functional.normalize(delta, mean, std)
    img = torch.mul(base, 1 - mask) + torch.mul(delta_norm, mask)
    return img


def get_data(device, encoder, loader, image_size, output_size):
    # output_size = encoder.out_size

    # output_size = getattr(models, arch)(
    #     weights=None
    # ).fc.in_features

    input_size = (3, image_size, image_size)
    xs = torch.empty(
        len(loader), loader.batch_size, *input_size, dtype=torch.float32, device=device
    )
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long, device=device)
    reps = torch.empty(
        len(loader), loader.batch_size, output_size, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # reps[i] = encoder.model(x)
            reps[i] = encoder(x)
            xs[i] = x
            ys[i] = y
    xs = xs.view(-1, *input_size)
    ys = ys.view(-1)
    reps = reps.view(-1, output_size)
    return reps.to("cpu"), xs.to("cpu"), ys.to("cpu")


class DatasetInit(data.Dataset):
    def __init__(self, train_probe_loader, transform, ratio):
        self.transform = transform
        # self.file_list = []

        train_probe_dataset = train_probe_loader.dataset

        self.original_length = len(train_probe_dataset)

        self.file_list = train_probe_dataset[:][
            :2
        ]  # tuple, image [500, 3, 32, 32] and label [500]

    def __getitem__(self, idx):
        image, target = self.file_list[0][idx], self.file_list[1][idx]
        image = self.transform(image)
        return image, target

    def __len__(self):
        return self.original_length


class DatasetEval(data.Dataset):
    def __init__(self, x, sample_size):
        x_indices = torch.randint(0, x.shape[0], (sample_size,))
        self.x = x[x_indices]

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


def eval_knn(device, encoder, loader, rep_center, y_center, target, output_size, k=1):
    rep_center, y_center = rep_center.to(device), y_center.to(device)

    with torch.no_grad():
        rep = torch.empty(
            (len(loader), loader.batch_size, output_size),
            dtype=torch.float,
            device=device,
        )
        for i, x in enumerate(loader):
            x = x.to(device)
            rep[i] = encoder(x)
        rep = rep.view((-1, output_size))
        d_t = torch.cdist(rep, rep_center)
        topk_t = torch.topk(d_t, k=k, dim=1, largest=False)
        labels_t = y_center[topk_t.indices]
        pred_t = torch.empty(rep.shape[0], device=device)
        for i in range(len(labels_t)):
            x = labels_t[i].unique(return_counts=True)
            pred_t[i] = x[0][x[1].argmax()]
        asr = (pred_t == target).float().mean().item()

    return asr


def trigger_inversion(args, model, poison):
    if args.method == "mocov2":
        backbone = copy.deepcopy(model.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model.backbone)

    backbone = backbone.eval()

    with torch.no_grad():
        transform = T.Compose(
            [
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
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=30).fit(
            rep
        )
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
                lam *= args.lam_multiplier_up

            elif lam != 0 and cost_down_counter >= args.patience:
                cost_down_counter = 0
                lam /= args.lam_multiplier_up

        reg_best_list[target] = reg_best if reg_best != torch.inf else 1

        os.makedirs(args.trigger_path, exist_ok=True)
        torch.save(
            {"mask": mask_best, "delta": delta_best},
            os.path.join(args.trigger_path, f"{target}.pth"),
        )


def get_scheduler(args, optimizer):
    m = [args.mitigate_epoches - a for a in args.drop]
    return MultiStepLR(optimizer, milestones=m, gamma=args.drop_gamma)


def trigger_mitigation(args):
    if args.method == "mocov2":
        backbone = copy.deepcopy(model.encoder_q)
        backbone.fc = nn.Sequential()
    else:
        backbone = copy.deepcopy(model.backbone)

    backbone_unlearn_trigger = copy.deepcopy(backbone)
    backbone_unlearn_trigger = (
        backbone_unlearn_trigger.train()
    )  # TODO: does it need to be set to eval mode at some point?
    backbone = backbone.eval()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, backbone_unlearn_trigger.parameters()),
        lr=3e-3,
        weight_decay=1e-6,
    )
    scheduler = get_scheduler(args, optimizer)
    eval_every = args.eval_every
    lr_warmup = 0
    torch.backends.cudnn.benchmark = True

    # TODO: fix
    dataloader = DataLoader(
        dataset=self.ds_train(),
        batch_size=128,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # generator=g,
    )

    for ep in range(args.mitigate_epoches):
        iters = len(dataloader)
        loss_ep = torch.empty((iters, 4))
        for n_iter, (samples, label, trigger) in enumerate(dataloader, start=1):
            samples = tuple(samples)

            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = 3e-3 * lr_scale
                lr_warmup += 1
            optimizer.zero_grad()

            # TODO: fix
            loss_1, _, _, loss_4 = model(samples, trigger, args.n_0, args.n_1, args.n_2)
            loss_sum = loss_1 * args.alpha_1 + loss_4 * args.alpha_4

            loss_sum.backward()

            optimizer.step()
            # loss_ep[n_iter] = torch.tensor([loss_1, loss_2, loss_3, loss_4])
            # TODO: fix
            model.step(ep / args.mitigate_epoches)

        scheduler.step()

        # TODO: fix
        if (ep + 1) % eval_every == 0:
            fname = f"{cfg.save_folder_root}/{cfg.exp_id}/{ep+1}.pt"
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            torch.save(backbone_unlearn_trigger.state_dict(), fname)
