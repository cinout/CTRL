import torch.nn.functional as F
from torch.utils import data
import torch


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
    delta_norm = F.normalize(delta, mean, std)
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
