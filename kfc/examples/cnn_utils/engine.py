import math
import sys
import torch
from tqdm import tqdm

sys.path.append('..')
from utils import Metric, accuracy

def train(epoch,
          model,
          optimizer, 
          preconditioner, 
          loss_func, 
          train_sampler, 
          train_loader, 
          args):
    # Set the model in training mode
    model.train()
    
    # Set the epoch for the train sampler
    train_sampler.set_epoch(epoch)
    
    # Initialize metrics for tracking train loss and accuracy
    train_loss = Metric('train_loss') 
    train_accuracy = Metric('train_accuracy')
    
    # Get the gradient scaler if available
    scaler = args.grad_scaler if 'grad_scaler' in args else None

    # Initialize a progress bar using tqdm
    with tqdm(total=len(train_loader),
              bar_format='{l_bar}{bar:10}{r_bar}',
              desc='Epoch {:3d}/{:3d}'.format(epoch, args.epochs),
              disable=not args.verbose) as t:
        # Iterate over batches in the train loader
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data and target to the GPU if available
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            # Reset gradients in the optimizer
            optimizer.zero_grad()

            # Split the data into mini-batches
            batch_idx = range(0, len(data), args.batch_size)
            for i in batch_idx:
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]

                # Perform forward pass and calculate loss
                if scaler is not None:
                    # Automatic mixed precision training
                    with torch.cuda.amp.autocast():
                        output = model(data_batch)
                        loss = loss_func(output, target_batch)
                else:
                    # Standard precision training
                    output = model(data_batch)
                    loss = loss_func(output, target_batch)
                
                # Normalize the loss by the number of batches per allreduce
                loss = loss / args.batches_per_allreduce

                # Update metrics without computing gradients
                with torch.no_grad():
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(output, target_batch))

                if i < batch_idx[-1]:
                    # Perform backward pass for intermediate mini-batches without synchronization
                    with model.no_sync():
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    # Perform backward pass and optimization step for the last mini-batch
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            # Update optimizer using gradient scaler if available
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Update the progress bar with relevant information
            t.set_postfix_str("loss: {:.4f}, acc: {:.2f}%, lr: {:.4f}".format(
                    train_loss.avg, 100*train_accuracy.avg,
                    optimizer.param_groups[0]['lr']))
            t.update(1)



def test(epoch, 
         model, 
         loss_func, 
         val_loader, 
         args):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              bar_format='{l_bar}{bar:10}|{postfix}',
              desc='             '.format(epoch, args.epochs),
              disable=not args.verbose) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))
                val_accuracy.update(accuracy(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str("\b\b val_loss: {:.4f}, val_acc: {:.2f}%".format(
                            val_loss.avg, 100*val_accuracy.avg),
                            refresh=False)

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        args.log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
