import math
import sys
import torch
from tqdm import tqdm
from cnn_utils.functions import   cross_entropy_loss

sys.path.append('..')
from utils import accuracy, Averagvalue

def train(epoch,
          model,
          optimizer, 
          preconditioner, 
          train_sampler, 
          train_loader,
          scheduler, 
          args):
    # # Set the model in training mode
    # model.train()
    
    # Set the epoch for the train sampler
    train_sampler.set_epoch(epoch)
    
    # Initialize metrics for tracking train loss and accuracy
    losses = Averagvalue() # Average loss value across batches
    epoch_loss = []  # List to store the loss for each epoch
    val_losses = Averagvalue()  # Average validation loss value across batches
    epoch_val_loss = []  # List to store the validation loss for each epoch
    counter = 0  # Counter to track iterations within an epoch

    # Initialize a progress bar using tqdm
    with tqdm(total=len(train_loader),
              bar_format='{l_bar}{bar:10}{r_bar}',
              desc='Epoch {:3d}/{:3d}'.format(epoch, args.epochs),
              disable=not args.verbose) as t:
        # Iterate over batches in the train loader
        for batch in train_loader:

            # Get data and label from the batch
            data, label, image_name = batch['data'], batch['label'], batch['id'][0]

            # Move data and label to the GPU if available
            if args.cuda:
                for key in data:
                    data[key] = data[key].cuda()
                label = label.cuda()

            image = data['image']

            # Perform forward pass and calculate loss
            outputs = model(data)

            ## loss
            if args.cuda:
                loss = torch.zeros(1).cuda()  # Initialize loss as zero on GPU
            else:
                loss = torch.zeros(1)  # Initialize loss as zero on CPU
            for o in outputs:
                loss = loss + cross_entropy_loss(o, label)  # Compute the cross-entropy loss for each output
            counter += 1
            loss = loss / args.itersize  # Average the loss across iterations
            loss.backward()  # Backpropagate the loss


            # SGD step
            if counter == args.itersize:
                optimizer.step()  # Update model parameters using the optimizer
                optimizer.zero_grad()  # Reset gradients
                counter = 0
                
                # Adjust learning rate
                scheduler.step()
                args.global_step += 1


            # Measure accuracy and record loss
            losses.update(loss.item(), image.size(0))  # Update the average loss value
            epoch_loss.append(loss.item())  # Append the loss to the list for the current epoch

            # Update the progress bar with relevant information
            t.set_postfix_str("loss: {:.4f}, lr: {:.4f}".format(
                    losses.avg,
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
