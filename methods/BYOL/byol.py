import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import CLModel
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar


class BYOL(CLModel):
    """
    Build a BYOL model. https://arxiv.org/abs/2006.07733
    """

    # def __init__(self, encoder_q, encoder_k, dim=4096, pred_dim=256, m=0.996):
    def __init__(self, args):
        """
        encoder_q: online network
        encoder_k: target network
        dim: feature dimension (default: 4096)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(BYOL, self).__init__(args)

        self.args = args

        # encoder_k = backbone_k

        # self.encoder_q = encoder_q
        # backbone_q = backbone
        self.backbone_k = self.model_generator()
        self.m = args.byol_m

        # projector

        # projector
        # encoder_dim = self.encoder_q.fc.weight.shape[1]
        self.projector_q = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.projector_k = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.predictor = nn.Sequential(
            nn.Linear(256, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256),
        )

        self.encoder_q = nn.Sequential(self.backbone, self.projector_q)

        self.encoder_k = nn.Sequential(self.backbone_k, self.projector_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        p1 = self.predictor(self.encoder_q(x1))  # NxC
        z2 = self.encoder_k(x2)  # NxC

        p2 = self.predictor(self.encoder_q(x2))  # NxC
        z1 = self.encoder_k(x1)  # NxC

        return p1, p2, z1, z2

    def negcos(self, p1, p2, z1, z2, mean=True):

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        if mean:
            standard_byol_loss = -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
            )
            if self.args.ssl_covariance_loss:
                # TODO: add regularisation loss (DONE)
                N, C = p1.shape
                off_diag_mask = ~torch.eye(C, dtype=bool)

                # p1
                p1 = p1 - p1.mean(dim=0)
                cov_p1 = (p1.T @ p1) / N  # C*C
                cov_p1_off_diagonal_elements = cov_p1[off_diag_mask]
                loss_p1 = torch.pow(cov_p1_off_diagonal_elements, 2).sum() / C

                # p2
                p2 = p2 - p2.mean(dim=0)
                cov_p2 = (p2.T @ p2) / N  # C*C
                cov_p2_off_diagonal_elements = cov_p2[off_diag_mask]
                loss_p2 = torch.pow(cov_p2_off_diagonal_elements, 2).sum() / C

                # z1
                z1 = z1.detach() - z1.detach().mean(dim=0)
                cov_z1 = (z1.T @ z1) / N  # C*C
                cov_z1_off_diagonal_elements = cov_z1[off_diag_mask]
                loss_z1 = torch.pow(cov_z1_off_diagonal_elements, 2).sum() / C

                # z2
                z2 = z2.detach() - z2.detach().mean(dim=0)
                cov_z2 = (z2.T @ z2) / N  # C*C
                cov_z2_off_diagonal_elements = cov_z2[off_diag_mask]
                loss_z2 = torch.pow(cov_z2_off_diagonal_elements, 2).sum() / C

                loss_covariance = loss_p1 + loss_p2 + loss_z1 + loss_z2

                return (
                    standard_byol_loss
                    + self.args.ssl_covariance_loss_w * loss_covariance
                )
            else:
                return standard_byol_loss
        else:
            # NOT USED
            return -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1)
                + F.cosine_similarity(p2, z1.detach(), dim=-1)
            )
