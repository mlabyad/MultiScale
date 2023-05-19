import sys
import kfac
import torch.optim as optim

sys.path.append('..')
from utils import create_lr_schedule

def get_optimizer(model, args, batch_first=True):

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.kfac_comm_method == 'comm-opt':
        comm_method=kfac.CommMethod.COMM_OPT
    elif args.kfac_comm_method == 'mem-opt':
        comm_method=kfac.CommMethod.MEM_OPT
    elif args.kfac_comm_method == 'hybrid-opt':
        comm_method=kfac.CommMethod.HYBRID_OPT
    else:
        raise ValueError('Unknwon KFAC Comm Method: {}'.format(
                args.kfac_comm_method))

    preconditioner = None

    lrs = create_lr_schedule(args.backend.size(), args.warmup_epochs, args.lr_decay)
    lr_scheduler = [optim.lr_scheduler.LambdaLR(optimizer, lrs)]

    return optimizer, preconditioner, lr_scheduler
