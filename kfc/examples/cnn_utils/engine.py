import math
import sys
import torch
from tqdm import tqdm
from cnn_utils.functions import   cross_entropy_loss
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
from os.path import join, split, isdir, isfile, splitext
from cnn_utils.functions import   cross_entropy_loss # sigmoid_cross_entropy_loss

sys.path.append('..')
from utils import accuracy, Averagvalue

def train(epoch,
          model,
          optimizer, 
          train_sampler, 
          train_loader,
          scheduler, 
          args,
          save_dir):

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
              disable=not args.verbose) as pbar:
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
            pbar.set_postfix(**{'loss (batch)': loss.item()})  # Update the progress bar with the current batch loss
            pbar.update(image.shape[0])  # Move the progress bar forward by the size of the current batch

            if (args.global_step >0) and (args.global_step % 500 ==0): #(self.n_dataset // (10 * self.batch_size)) == 0:
                    ## logging
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        args.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), args.global_step)
                        args.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), args.global_step)

                    args.writer.add_images('images', image, args.global_step)
                    args.writer.add_images('masks/true', label, args.global_step)
                    args.writer.add_images('masks/pred', outputs[-1] > 0.5, args.global_step)

                    outputs.append(label)
                    outputs.append(image)

                    dev_checkpoint(save_dir=join(save_dir, f'training-epoch-{epoch+1}-record'),
                                i=args.global_step, epoch=epoch, image_name=image_name, outputs= outputs)

        save_state(epoch, save_path=join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))
        args.writer.add_scalar('Loss_avg', losses.avg, epoch+1)
        # Update the training loss and accuracy for the current epoch
        args.train_loss.append(losses.avg)
        args.train_loss_detail += epoch_loss
        if val_losses.count>0:
            args.writer.add_scalar('Val_Loss_avg', val_losses.avg, epoch+1)
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], args.global_step)


def test(epoch,
          model,
          dev_loader,
          args,
          save_dir):
    print("Running test ========= >")
    model.eval()
    for idx, batch in enumerate(dev_loader):

        data, label, image_name= batch['data'], batch['label'], batch['id'][0]

        if args.cuda:
            for key in data:
                data[key] = data[key].cuda()
            label = label.cuda()

        _, _, H, W = data['image'].shape

        if torch.cuda.is_available():
            for key in data:
                data[key]=data[key].cuda()
            label = label.cuda()

        with torch.no_grad():
            outputs = model(data)

        outputs.append(1-outputs[-1])
        outputs.append(label)
        dev_checkpoint(save_dir, -1, epoch, image_name, outputs)


def save_state(self, epoch, save_path='checkpoint.pth'):
        torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, save_path)



##=========================== train_split func

def dev_checkpoint(save_dir, i, epoch, image_name, outputs):
    # display and logging
    if not isdir(save_dir):
        os.makedirs(save_dir)
    outs=[]
    for o in outputs:
        outs.append(tensor2image(o))
    if len(outs[-1].shape)==3:
        outs[-1]=outs[-1][0,:,:] #if RGB, show one layer only
    if i==-1:
        output_name=f"{image_name}.jpg"
    else:
        output_name=f"global_step-{i}-{image_name}.jpg"
    out=cv2.hconcat(outs) # if gray
    cv2.imwrite(join(save_dir, output_name), out)

def tensor2image(image):
            result = torch.squeeze(image.detach()).cpu().numpy()
            result = (result * 255).astype(np.uint8, copy=False)
            #(torch.squeeze(o.detach()).cpu().numpy()*255).astype(np.uint8, copy=False)
            return result

