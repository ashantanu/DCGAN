import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import utils

from Generator import Generator
from Discriminator import Discriminator

#initial steps
config_file = "config.yml"
config = utils.load_config(config_file)
utils.set_manual_seed(config)
dataset = utils.load_transformed_dataset(config)
dataloader = utils.get_dataloader(config,dataset)

gen = Generator(config)
dis = Discriminator(config)
gen.apply(utils.init_weights)
dis.apply(utils.init_weights)

gen_optimizer = torch.optim.Adam(params=gen.parameters(), 
                            lr=config['lr'],
                            betas=[config['beta1'],config['beta2']])
dis_optimizer = torch.optim.Adam(params=dis.parameters(), 
                            lr=config['lr'],
                            betas=[config['beta1'],config['beta2']])

criterion = torch.nn.BCELoss()
fixed_latent = torch.randn(16,config['len_z'])

dis_loss = []
gen_loss = []
generated_imgs = []
iteration = 0
#training
for epoch in range(config['epochs']):
    iterator = iter(dataloader)
    dataloader_flag = True
    while(dataloader_flag):
        for _ in range(config['discriminator_steps']):
            dis_optimizer.zero_grad()

            #sample mini-batch
            z = torch.randn(config['batch_size'],config['len_z'])
            fake_images = gen(z)

            #get images from dataloader via iterator
            try:
                imgs, _ = next(iterator)
            except:
                dataloader_flag = False
                break

            #compute loss
            loss_true_imgs = criterion(dis(imgs).view(-1),torch.ones(imgs.shape[0]))
            loss_fake_imgs = criterion(dis(fake_images.detach()).view(-1),torch.zeros(z.shape[0]))

            #update params
            loss_true_imgs.backward()
            loss_fake_imgs.backward()
            total_error = loss_fake_imgs+loss_true_imgs
            dis_optimizer.step()
        
        #generator step
        gen_optimizer.zero_grad()
        z = torch.randn(config['batch_size'],config['len_z'])   #sample mini-batch
        loss_gen = criterion(dis(gen(z)).view(-1),torch.ones(z.shape[0]))    #compute loss

        #update params
        loss_gen.backward()
        gen_optimizer.step()

        iteration+=1

        if(iteration==21):
            break
        #log things
        if(iteration%10)==0:
            dis_loss.append(loss_true_imgs.mean().item()+loss_fake_imgs.mean().item())
            gen_loss.append(loss_gen.mean().item())

            if(epoch%10==0) or iteration%10==0:
                generated_imgs.append(gen(fixed_latent).detach())    #generate image
                print("Loss epoch:%d, Dis Loss:%.4f, Gen Loss:%.4f",epoch,dis_loss[-1],gen_loss[-1])


#plot errors
utils.save_loss_plot(gen_loss,dis_loss)

#plot generated images
utils.save_result_images(next(iter(dataloader))[0],generated_imgs[-1],4)

#save generated images so see what happened
torch.save(generated_imgs,"gen_imgs_array.pt")