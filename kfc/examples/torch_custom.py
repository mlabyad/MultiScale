import argparse
import time
import os
import sys
import datetime
import kfac
import torch
import torch.distributed as dist

from msnet import msNet
import cnn_utils.datasets as datasets
import cnn_utils.engine as engine
import cnn_utils.optimizers as optimizers
from pathlib import Path

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint

try:
    from torch.cuda.amp import autocast, GradScaler
    TORCH_FP16 = True
except:
    TORCH_FP16 = False

def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--data-dir', type=str, default='/tmp/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--log-dir', default='./logs/torch_cifar10',
                        help='TensorBoard/checkpoint directory')
    parser.add_argument('--checkpoint-format', default='checkpoint_{epoch}.pth.tar',
                        help='checkpoint file format')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use torch.cuda.amp for fp16 training (default: false)')

    # Training settings
    parser.add_argument('--model', type=str, default='resnet32',
                        help='ResNet model to use [20, 32, 56]')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='input batch size for validation (default: 128)')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--base-lr', type=float, default=1e-06, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', nargs='+', type=int, default=[35, 75, 90],
                        help='epoch intervals to decay lr (default: [35, 75, 90])')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='epochs between checkpoints')
    parser.add_argument('--stepsize', type=int, default=1e4,
                        help='epochs between checkpoints')
    parser.add_argument('--gamma', type=int, default=0.1,
                        help='epochs between checkpoints')

    # KFAC Parameters
    parser.add_argument('--kfac-update-freq', type=int, default=10,
                        help='iters between kfac inv ops (0 disables kfac) (default: 10)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=0,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--kfac-update-freq-alpha', type=float, default=10,
                        help='KFAC update freq multiplier (default: 10)')
    parser.add_argument('--kfac-update-freq-decay', nargs='+', type=int, default=None,
                        help='KFAC update freq decay schedule (default None)')
    parser.add_argument('--use-inv-kfac', action='store_true', default=False,
                        help='Use inverse KFAC update instead of eigen (default False)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.003,
                        help='KFAC damping factor (defaultL 0.003)')
    parser.add_argument('--damping-alpha', type=float, default=0.5,
                        help='KFAC damping decay factor (default: 0.5)')
    parser.add_argument('--damping-decay', nargs='+', type=int, default=None,
                        help='KFAC damping decay schedule (default None)')
    parser.add_argument('--kl-clip', type=float, default=0.001,
                        help='KL clip (default: 0.001)')
    parser.add_argument('--skip-layers', nargs='+', type=str, default=[],
                        help='Layer types to ignore registering with KFAC (default: [])')
    parser.add_argument('--coallocate-layer-factors', action='store_true', default=True,
                        help='Compute A and G for a single layer on the same worker. ')
    parser.add_argument('--kfac-comm-method', type=str, default='comm-opt',
                        help='KFAC communication optimization strategy. One of comm-opt, '
                             'mem-opt, or hybrid_opt. (default: comm-opt)')
    parser.add_argument('--kfac-grad-worker-fraction', type=float, default=0.25,
                        help='Fraction of workers to compute the gradients '
                             'when using HYBRID_OPT (default: 0.25)')
    
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for distribute training (default: nccl)')
    # Set automatically by torch distributed launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize the distributed process group
    torch.distributed.init_process_group(backend=args.backend, init_method='env://')
    
    # Initialize the communication backend
    kfac.comm.init_comm_backend() 

    # Set CUDA device if enabled
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Print rank, world size, and device IDs
    print('rank = {}, world_size = {}, device_ids = {}'.format(
            torch.distributed.get_rank(), torch.distributed.get_world_size(),
            args.local_rank))

    # Update args with backend, base_lr, verbose, and horovod properties
    args.backend = kfac.comm.backend
    args.base_lr = args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    args.verbose = True if dist.get_rank() == 0 else False
    args.horovod = False

    args.root = Path("/scratch1/99999/malb23/ASC22050/SR_Dataset_v1/cresis-data")
    args.trainlist = Path("../data/train.lst")
    args.devlist = Path("../data/dev.lst")
    print(args)

    # Get data loaders for training and validation datasets
    train_sampler, train_loader, _, val_loader = datasets.get_cifar(args)

    # Instantiate the model
    model = model=msNet()

    # Set the device (CPU or CUDA) for the model
    device = 'cpu' if not args.cuda else 'cuda' 
    model.to(device)

    # Wrap the model with DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, 
            device_ids=[args.local_rank])

    # Create log directory and set checkpoint format
    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)

    # Create a SummaryWriter for logging if verbose is True
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    # Initialize resume_from_epoch to 0
    args.resume_from_epoch = 0

    # Find the latest checkpoint and update resume_from_epoch if found
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            args.resume_from_epoch = try_epoch
            break
    
    # Initialize scaler for mixed precision training
    scaler = None
    if args.fp16:
        # Check if torch.cuda.amp fp16 training is supported
        if not TORCH_FP16:
            raise ValueError('The installed version of torch does not '
                             'support torch.cuda.amp fp16 training. This '
                             'requires torch version >= 1.16')
        scaler = GradScaler()
    args.grad_scaler = scaler

    # Get optimizer, preconditioner, and lr_schedules
    optimizer, preconditioner, lr_schedules = optimizers.get_optimizer(model, args)

    # # Set the loss function
    # loss_func = torch.nn.CrossEntropyLoss()

    # Load model, optimizer, preconditioner, and schedulers from checkpoint if resuming from a previous epoch
    if args.resume_from_epoch > 0:
        # Get the file path of the checkpoint for the specified epoch
        filepath = args.checkpoint_format.format(epoch=args.resume_from_epoch)
        
        # Define the mapping for loading the checkpoint on the appropriate device
        map_location = {'cuda:0': 'cuda:{}'.format(args.local_rank)}
        
        # Load the checkpoint from the specified file path
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load the model's state dictionary from the checkpoint
        model.module.load_state_dict(checkpoint['model'])
        
        # Load the optimizer's state dictionary from the checkpoint
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load schedulers' state dictionaries if the checkpoint contains a list of schedulers
        if isinstance(checkpoint['schedulers'], list):
            for sched, state in zip(lr_schedules, checkpoint['schedulers']):
                sched.load_state_dict(state)
        
        # Load preconditioner's state dictionary if the checkpoint contains a preconditioner and preconditioner is not None
        if (checkpoint['preconditioner'] is not None and 
                preconditioner is not None):
            preconditioner.load_state_dict(checkpoint['preconditioner'])

    # Start time
    start = time.time()

    # Training loop
    args.global_step = 0
    for epoch in range(args.resume_from_epoch + 1, args.epochs + 1):
        # Perform training step
        engine.train(epoch, model, optimizer, preconditioner,
                    train_sampler, train_loader, lr_schedules, args)
        
        # Evaluate model on validation set
        engine.test(epoch, model, val_loader, args)
        
        # Save checkpoint if epoch is a multiple of checkpoint_freq and rank is 0
        if (epoch > 0 and epoch % args.checkpoint_freq == 0 and 
                dist.get_rank() == 0):
            # Note: save model.module because model may be a Distributed wrapper
            # Saving the underlying model is more generic
            save_checkpoint(model.module, optimizer, preconditioner, lr_schedules,
                            args.checkpoint_format.format(epoch=epoch))

    # Print total training time if verbose is True
    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))



if __name__ == '__main__': 
    main()
