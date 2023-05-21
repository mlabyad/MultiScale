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
import modules.data_loader as data_loader
from modules.trainer import Trainer, Network
from pathlib import Path


def parse_args():
    # General settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/scratch1/99999/malb23/ASC22050/SR_Dataset_v1/cresis-data', metavar='D',
                        help='directory to download cifar10 dataset to')
    parser.add_argument('--trainlist', type=str, default='./data/train.lst')
    parser.add_argument('--devlist', type=str, default='./data/dev.lst')
    parser.add_argument('--tmp', type=str, default='../tmp/tag')
    parser.add_argument('--log-dir', type=str, default='../logs/tag')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 15)')
    parser.add_argument('--val-batch-size', type=int, default=1, help='input batch size for validation (default: 1)')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--lr', type=float, default=1e-06, metavar='LR', help='base learning rate (default: 1e-06)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--stepsize', type=int, default=1e4, help='epochs between checkpoints')
    parser.add_argument('--gamma', type=int, default=0.1, help='epochs between checkpoints')
    parser.add_argument('--weight-decay', type=int, default=0.0002)
    parser.add_argument('--itersize', type=int, default=10)
    parser.add_argument('--max-epoch', type=int, default=15)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--weights-init-on', type=bool, default=False)

    # KFAC Parameters
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distribute training (default: nccl)')
    # Set automatically by torch distributed launch
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.root = Path(args.root)
    args.trainlist = Path(args.trainlist)
    args.devlist = Path(args.devlist)
    tag = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # tag = 'last_experiment'
    args.tmp = Path(args.tmp.replace('tag', tag))
    args.log_dir = Path(args.log_dir.replace('tag', tag))

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

    # Get data loaders for training and validation datasets
    train_sampler, train_loader, _, dev_loader = data_loader.get_data(args)

    # define network
    net=Network(args, model=msNet())

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Start time
    start = time.time()

    # define trainer
    trainer = Trainer(args, net, train_sampler=train_sampler, train_loader=train_loader)
    for epoch in range(args.start_epoch, args.max_epoch):
        ## initial log (optional:sample36)
        if (epoch == 0) and (args.devlist is not None):
            print("Performing initial testing...")
            trainer.test(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'), epoch=epoch)
    
        ## training
        trainer.train(save_dir = args.tmp, epoch=epoch)
    
        ## dev check (optional:sample36)
        if args.devlist is not None:
            trainer.test(dev_loader=dev_loader, save_dir = join(args.tmp, f'testing-record-epoch-{epoch+1}'), epoch=epoch)

    # Print total training time if verbose is True
    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))


if __name__ == '__main__': 
    main()
