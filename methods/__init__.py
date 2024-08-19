from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL


def set_model(args):
    if args.method == "simclr":
        return SimCLRModel(args)
    elif args.method == "byol":
        return BYOL(args)
    # TODO: add MoCoV2
    else:
        raise NotImplementedError
