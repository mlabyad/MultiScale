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
import modules.trainer as trainer
import modules.optimizers as optimizers
from pathlib import Path

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--base-lr', type=float, default=1e-06, metavar='LR',
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

    # Update args with backend, base_lr, verbose, and horovod properties
    args.backend = kfac.comm.backend
    args.base_lr = args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    args.verbose = True if dist.get_rank() == 0 else False
    args.horovod = False

    args.root = Path("/scratch1/99999/malb23/ASC22050/SR_Dataset_v1/cresis-data")
    args.trainlist = Path("../data/train.lst")
    args.devlist = Path("../data/dev.lst")
    tag = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.tmp = Path(f'../tmp/{tag}')
    args.itersize = 10
    args.resume_path = None

    # Get data loaders for training and validation datasets
    train_sampler, train_loader, _, dev_loader = datasets.get_cifar(args)

    # Instantiate the model
    model = model=msNet()

    # Set the device (CPU or CUDA) for the model
    device = 'cpu' if not args.cuda else 'cuda' 
    model.to(device)

    # Wrap the model with DistributedDataParallel for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    if args.resume_path is not None:
        trainer.resume(model=model, resume_path=args.resume_path)

    # Create log directory and set checkpoint format
    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)

    # Create a SummaryWriter for logging if verbose is True
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None

    # Initialize resume_from_epoch to 0
    args.resume_from_epoch = 0

    # # Find the latest checkpoint and update resume_from_epoch if found
    # for try_epoch in range(args.epochs, 0, -1):
    #     if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
    #         args.resume_from_epoch = try_epoch
    #         break
    
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
    args.train_loss = []
    args.train_loss_detail = []
    args.writer = SummaryWriter(args.log_dir) if args.verbose else None
    args.max_epoch = 2
    args.start_epoch = 0
    args.n_train = len(train_loader)
    for epoch in range(args.start_epoch, args.max_epoch):
        ## initial log (optional:sample36)
        if (epoch == 0) and (args.devlist is not None):
            print("Performing initial testing...")
            trainer.test(epoch, model, dev_loader, args, save_dir = join(args.tmp, 'testing-record-0-initial'))
        # Perform training step
        trainer.train(epoch, model, optimizer, train_sampler, train_loader, lr_schedules, args, save_dir = args.tmp)
        
        # Evaluate model on validation set
        if args.devlist is not None:
            trainer.test(epoch, model, dev_loader, args, save_dir = join(args.tmp, f'testing-record-epoch-{epoch+1}'))

    # Print total training time if verbose is True
    if args.verbose:
        print('\nTraining time: {}'.format(datetime.timedelta(seconds=time.time() - start)))



if __name__ == '__main__': 
    main()
