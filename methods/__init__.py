from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL
from .MoCoV2.mocov2 import MoCo
import torchvision.models as models


def set_model(args):
    if args.method == "simclr":
        return SimCLRModel(args)
    elif args.method == "byol":
        return BYOL(args)
    elif args.method == "mocov2":
        return MoCo(
            models.__dict__[args.arch],
            dim=128,
            K=65536,
            m=0.999,
            contr_tau=0.2,
            align_alpha=2,
            unif_t=3,
            unif_intra_batch=True,
            mlp=True,
        )
    else:
        raise NotImplementedError
