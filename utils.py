import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_config(file_name):
    config=yaml.safe_load(open(file_name))
    return config

def set_manual_seed(config):
    if 'reproduce' in config and config['reproduce']==True:
        seed = 0 if 'seed' not in config else config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

def load_transformed_dataset(config):
    h = config['img_h']
    w = config['img_w']
    mean = config['input_normalise_mean']
    std = config['input_normalise_std']

    dataset = ImageFolder(config['data_path'], 
                        transform=transforms.Compose([
                            transforms.Resize(h),#to maintain aspect ratio
                            transforms.CenterCrop((h,w)),
                            transforms.ToTensor(),#normalize takes a tensor
                            transforms.Normalize((mean,mean,mean),(std,std,std))
                            ]))
    return dataset

def get_dataloader(config, dataset):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=config['shuffle'], num_workers=config['num_workers'])
    return dataloader

def init_weights(m):
    '''
    if type(m) in [ torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Linear ]:
        torch.nn.init.normal_(m.weight,0.0,0.02)
    if type(m) == torch.nn.BatchNorm2d: #copied this from tutorial. Need to figure out the logic
        torch.nn.init.normal_(m.weight,1,0.02)
        torch.nn.init.constant_(m.bias,0)
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_loss_plot(g_loss,d_loss):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss,label="G")
    plt.plot(d_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss.png")

def save_result_images(real_images,fake_images,nrow,config):
    mean = config['input_normalise_mean']
    std = config['input_normalise_std']
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    grid_real = vutils.make_grid(real_images*std+mean,nrow=nrow)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(grid_real.permute(1,2,0))
    plt.subplot(1,2,2)
    grid_fake = vutils.make_grid(fake_images*std+mean,nrow=nrow)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(grid_fake.permute(1,2,0))
    plt.savefig("./generated_images.png",dpi=300)