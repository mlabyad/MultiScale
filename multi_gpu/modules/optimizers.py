import sys
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append('..')

def get_optimizer(model, args, batch_first=True):

    tuned_lrs=tune_lrs(model, args.base_lr, args.weight_decay)

    optimizer = optim.SGD(
        tuned_lrs,
        lr=args.base_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    preconditioner = None

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    return optimizer, preconditioner, scheduler


##========================== adjusting lrs

def tune_lrs(model, lr, weight_decay):

    bias_params= [param for name,param in list(model.named_parameters()) if name.find('bias')!=-1]
    weight_params= [param for name,param in list(model.named_parameters()) if name.find('weight')!=-1]

    if len(weight_params)==19:
        down1_4_weights , down1_4_bias  = weight_params[0:10]  , bias_params[0:10]
        down5_weights   , down5_bias    = weight_params[10:13] , bias_params[10:13]
        up1_5_weights    , up1_5_bias     = weight_params[13:18] , bias_params[13:18]
        fuse_weights , fuse_bias =weight_params[-1] , bias_params[-1]
        
        tuned_lrs=[
        {'params': down1_4_weights, 'lr': lr*1    , 'weight_decay': weight_decay},
        {'params': down1_4_bias,    'lr': lr*2    , 'weight_decay': 0.},
        {'params': down5_weights,   'lr': lr*100  , 'weight_decay': weight_decay},
        {'params': down5_bias,      'lr': lr*200  , 'weight_upecay': 0.},
        {'params': up1_5_weights,    'lr': lr*0.01 , 'weight_decay': weight_decay},
        {'params': up1_5_bias,       'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': fuse_weights,    'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': fuse_bias ,      'lr': lr*0.002, 'weight_decay': 0.},
        ]

    elif len(weight_params)==32: #bn
        down1_4_weights , down1_4_bias  = weight_params[0:20]  , bias_params[0:20]
        down5_weights   , down5_bias    = weight_params[20:26] , bias_params[20:26]
        up1_5_weights    , up1_5_bias     = weight_params[26:31] , bias_params[26:31]
        fuse_weights , fuse_bias =weight_params[-1] , bias_params[-1]
        
        tuned_lrs=[
        {'params': down1_4_weights, 'lr': lr*1    , 'weight_decay': weight_decay},
        {'params': down1_4_bias,    'lr': lr*2    , 'weight_decay': 0.},
        {'params': down5_weights,   'lr': lr*100  , 'weight_decay': weight_decay},
        {'params': down5_bias,      'lr': lr*200  , 'weight_upecay': 0.},
        {'params': up1_5_weights,    'lr': lr*0.01 , 'weight_decay': weight_decay},
        {'params': up1_5_bias,       'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': fuse_weights,    'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': fuse_bias ,      'lr': lr*0.002, 'weight_decay': 0.},
        ]
    else:
        print('Warning in tune_lrs')
        return model.parameters()

    return  tuned_lrs