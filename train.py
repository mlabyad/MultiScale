import argparse
import time
import os
import sys
import datetime
import kfac
import torch
import torch.distributed as dist
from os.path import join, isdir
from msnet import msNet
import modules.datasets as datasets
from modules.trainer import Trainer, Network
from pathlib import Path


def parse_args():
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--data-dir', type=str, default='/tmp/cifar10', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='use torch.cuda.amp for fp16 training (default: false)')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='input batch size for validation (default: 128)')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--lr', type=float, default=1e-06, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                        help='number of warmup epochs (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--stepsize', type=int, default=1e4,
                        help='epochs between checkpoints')
    parser.add_argument('--gamma', type=int, default=0.1,
                        help='epochs between checkpoints')

    # KFAC Parameters
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

    # Update args with backend, lr, verbose, and horovod properties
    args.backend = kfac.comm.backend
    args.lr = args.lr * dist.get_world_size() * args.batches_per_allreduce
    args.verbose = True if dist.get_rank() == 0 else False

    args.root = Path("/scratch1/99999/malb23/ASC22050/SR_Dataset_v1/cresis-data")
    args.trainlist = Path("./data/train.lst")
    args.devlist = Path("./data/dev.lst")
    # tag = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = 'last_experiment'
    args.tmp = Path(f'../tmp/{tag}')
    args.log_dir = Path(f'../logs/{tag}')
    args.itersize = 10
    args.max_epoch = 1
    args.start_epoch = 0
    args.resume_path = None
    args.weights_init_on = None

    # Get data loaders for training and validation datasets
    train_sampler, train_loader, _, dev_loader = datasets.get_cifar(args)

    # Instantiate the model
    net = Network(args, model=msNet())

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Start time
    start = time.time()

    # Training loop
    trainer = Trainer(args, net, train_sampler=train_sampler, train_loader=train_loader)
    for epoch in range(args.start_epoch, args.max_epoch):
        ## initial log (optional:sample36)
        if (epoch == 0) and (args.devlist is not None):
            print("Performing initial testing...")
            trainer.test(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'), epoch=epoch)
        # Perform training step
        trainer.train(save_dir = args.tmp, epoch=epoch)
        
        # Evaluate model on validation set
        if args.devlist is not None:
            trainer.test(dev_loader=dev_loader,save_dir = join(args.tmp, f'testing-record-epoch-{epoch+1}'), epoch=epoch)

    # Print total training time if verbose is True
    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))



if __name__ == '__main__': 
    main()
