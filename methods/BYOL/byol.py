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

        # TODO: remove these
        print("p1.shape", p1.shape)
        print("p2.shape", p2.shape)
        print("z1.shape", z1.shape)
        print("z2.shape", z2.shape)

        if mean:
            # NOT USED
            return -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
            )
        else:
            standard_byol_loss = -0.5 * (
                F.cosine_similarity(p1, z2.detach(), dim=-1)
                + F.cosine_similarity(p2, z1.detach(), dim=-1)
            )

            if self.args.ssl_covariance_loss:
                # TODO: add regularisation loss

                pass
            else:

                return standard_byol_loss
