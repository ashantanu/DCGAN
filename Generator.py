import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from  torch.nn import Sequential, ConvTranspose2d, UpsamplingBilinear2d, \
    BatchNorm2d, ReLU, Tanh, Linear
from copy import copy 

import utils

logging.basicConfig(filename="./log_generator.txt", filemode='w',level=logging.INFO)
class Generator(torch.nn.Module):
    def __init__(self,config):
        super(Generator,self).__init__()

        self.parse_config(config)
        self.generator = Sequential()
        
        #first layer. Project and reshape noise
        self.c_latent = self.g_feature_size * (2**(self.g_layers-1))
        self.latent_hw = int(self.img_h/(2**self.g_layers))
        self.latent_reshape = Linear(self.c_input, self.c_latent * self.latent_hw * self.latent_hw)
        c_input = self.c_latent
        #will have to reshape this into a (c_latent, latent_hw, latent_hw) tensor

        layer_number = 1
        for i in range(self.g_layers-2,-1,-1):
            c_layer = int(self.g_feature_size *(2**i))
            self.generator.add_module('TConv'+str(layer_number),ConvTranspose2d(c_input,c_layer,self.kernel_size, self.stride,self.g_input_pad,output_padding=self.g_output_pad))
            self.generator.add_module('BN'+str(layer_number),BatchNorm2d(c_layer))
            self.generator.add_module('ReLU'+str(layer_number),ReLU())
            c_input = copy(c_layer)
            layer_number+=1

        #final image layer
        self.generator.add_module('F_TConv1',ConvTranspose2d(c_input,self.img_c,self.kernel_size, self.stride, self.g_input_pad, output_padding=self.g_output_pad))
        self.generator.add_module('F_BN1',BatchNorm2d(self.img_c))
        self.generator.add_module('F_Tanh',Tanh())

    
    def parse_config(self, config):
        self.g_feature_size=config['g_feature_size']
        self.g_layers = config['g_layers']
        self.len_z=config['len_z']
        self.img_h=config['img_h']
        self.img_w=config['img_w']
        self.img_c=config['img_c']
        self.c_input = config['len_z']
        self.stride = config['g_stride']
        self.kernel_size = config['g_kernel_size']
        self.g_input_pad = config['g_input_pad']
        self.g_output_pad = config['g_output_pad']
    
    def forward(self,z):
        logging.info("Input Shape = " + str(z.shape))
        z_projected = self.latent_reshape(z)
        z_reshape = z_projected.reshape(-1, self.c_latent, self.latent_hw, self.latent_hw)
        logging.info("Reshaped Latent Shape = " + str(z_reshape.shape))
        logging.info(self.generator)
        generated_image = self.generator(z_reshape)

        return generated_image