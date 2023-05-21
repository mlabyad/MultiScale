import os
import torch
from torch.utils import data as D
from os.path import join, splitext, basename #,split, abspath, splitext, split, isdir, isfile
import numpy as np
from modules.transforms import Fliplr, Rescale_byrate
from torch.utils.data import DataLoader, ConcatDataset
import cv2
import os
import pandas as pd


class SnowData(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    def __init__(self, root, lst, train=True, transform=None,  wt =None):
        self.df=pd.read_csv(lst, names=['data'])
        self.root=root #os.path.abspath(root)
        self.transform=transform
        self.train=train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        img_abspath= join(self.root, self.df['data'][index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

               # Edge Maps (binary files)


        img=cv2.imread(img_abspath,0)
        if self.transform:
            img=self.transform(img)
                
            #### will be added later
        # if self.wt is not None:
        #     data=get_wt(img, self.wt , mode='periodic', level=4)
        # else:
        #     data={}
        data={}
        img = prepare_img(img)
        data['image']=img
        
         #img=img[0,:,:]*np.ones(1, dtype=np.float32)[None, None, :]
        (data_id, _) = splitext(basename(img_abspath))
        if self.train:
            ct_abspath=img_abspath.replace('data_','layer_binary_')
            assert os.path.isfile(ct_abspath), "file  {}. doesn't exist.".format(ct_abspath)
            ctour=cv2.imread(ct_abspath, cv2.IMREAD_GRAYSCALE)
            if self.transform:
                ctour=self.transform(ctour)
            ctour= prepare_ctour(ctour)
           
    
            return {'data': data, 'label': ctour, 'id': data_id}
        else:    
            return {'data': data,  'id': data_id}            


def prepare_img(img):
        img=np.array(img, dtype=np.float32)
        #img=np.expand_dims(img,axis=2)
        (R,G,B)=(104.00698793,116.66876762,122.67891434)
        img -= np.array((0.299*R + 0.587*G + 0.114*B))
        #img=img*np.ones(1, dtype=np.float32)[None, None, :]
        #img=img.transpose(2,0,1)
        return np.expand_dims(img,axis=0)    


def prepare_ctour(ctour):
        #ctour=np.array(ctour, dtype=np.float32)
        ctour = (ctour > 0 ).astype(np.float32)
        return np.expand_dims(ctour,axis=0)


def prepare_w(img):
        img=np.array(img, dtype=np.float32)
        img=np.expand_dims(img,axis=0)
        return img


def get_data(args):
    ds=[SnowData(root=args.root,lst=args.trainlist),
    SnowData(root=args.root,lst=args.trainlist, transform=Rescale_byrate(.75)),
    SnowData(root=args.root,lst=args.trainlist,transform=Rescale_byrate(.5)),
    SnowData(root=args.root,lst=args.trainlist,transform=Rescale_byrate(.25)),
    SnowData(root=args.root,lst=args.trainlist, transform=Fliplr())
    ]
    train_dataset=ConcatDataset(ds)

    test_dataset=SnowData(root=args.root,lst=args.devlist)
    
    return make_sampler_and_loader(args, train_dataset, test_dataset)


def make_sampler_and_loader(args, train_dataset, val_dataset):
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.backend.size(), rank=args.backend.rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size * args.batches_per_allreduce,
            sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.backend.size(), rank=args.backend.rank())
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.val_batch_size,
            sampler=val_sampler, **kwargs)

    return train_sampler, train_loader, val_sampler, val_loader
