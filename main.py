import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

import utils
from Generator import Generator
from Discriminator import Discriminator
import torchvision.utils as vutils

#initial steps
config_file = "config.yml"
config = utils.load_config(config_file)
utils.set_manual_seed(config)
dataset = utils.load_transformed_dataset(config)
dataloader = utils.get_dataloader(config,dataset)
device = torch.device("cuda:0" if (torch.cuda.is_available() and config['ngpu'] > 0) else "cpu")
print("Using device : ", device)

#initialize models
gen = Generator(config).to(device)
dis = Discriminator(config).to(device)
gen.apply(utils.init_weights)
dis.apply(utils.init_weights)

#setup optimizers
gen_optimizer = torch.optim.Adam(params=gen.parameters(), 
                            lr=config['lr'],
                            betas=[config['beta1'],config['beta2']])
dis_optimizer = torch.optim.Adam(params=dis.parameters(), 
                            lr=config['lr'],
                            betas=[config['beta1'],config['beta2']])

criterion = torch.nn.BCELoss()
fixed_latent = torch.randn(16,config['len_z'],1,1,device=device)

dis_loss = []
gen_loss = []
generated_imgs = []
iteration = 0

#load parameters
if(config['load_params'] and os.path.isfile("./gen_params.pth.tar")):
    print("loading params...")
    gen.load_state_dict(torch.load("./gen_params.pth.tar"))
    dis.load_state_dict(torch.load("./dis_params.pth.tar"))
    gen_optimizer.load_state_dict(torch.load("./gen_optimizer_state.pth.tar"))
    dis_optimizer.load_state_dict(torch.load("./dis_optimizer_state.pth.tar"))
    generated_imgs = torch.load("gen_imgs_array.pt")
    print("loaded params.")

#training
start_time = time.time()
for epoch in range(config['epochs']):
    iterator = iter(dataloader)
    dataloader_flag = True
    while(dataloader_flag):
        for _ in range(config['discriminator_steps']):
            dis.zero_grad()
            gen.zero_grad()
            dis_optimizer.zero_grad()

            #sample mini-batch
            z = torch.randn(config['batch_size'],config['len_z'],1,1,device=device)

            #get images from dataloader via iterator
            try:
                imgs, _ = next(iterator)
                imgs = imgs.to(device)
            except:
                dataloader_flag = False
                break

            #compute loss
            loss_true_imgs = criterion(dis(imgs).view(-1),torch.ones(imgs.shape[0],device=device))
            loss_true_imgs.backward()
            fake_images = gen(z)    
            loss_fake_imgs = criterion(dis(fake_images.detach()).view(-1),torch.zeros(z.shape[0],device=device))
            loss_fake_imgs.backward()

            total_error = loss_fake_imgs+loss_true_imgs
            dis_optimizer.step()
        
        #generator step
        for _ in range(config['generator_steps']):
            if(dataloader_flag==False):
                break
            gen.zero_grad()
            dis.zero_grad()
            dis_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            #z = torch.randn(config['batch_size'],config['len_z'])   #sample mini-batch
            loss_gen = criterion(dis(fake_images).view(-1),torch.ones(z.shape[0],device=device))    #compute loss

            #update params
            loss_gen.backward()
            gen_optimizer.step()

        iteration+=1
        
        #log and save variable, losses and generated images
        if(iteration%50)==0:
            elapsed_time = time.time()-start_time
            dis_loss.append(total_error.mean().item())
            gen_loss.append(loss_gen.mean().item())

            if(iteration%100==0):
                with torch.no_grad():
                    generated_imgs.append(gen(fixed_latent).detach())    #generate image
                    torch.save(generated_imgs,"gen_imgs_array.pt")

            print("Iteration:%d, Dis Loss:%.4f, Gen Loss:%.4f, time elapsed:%.4f"%(iteration,dis_loss[-1],gen_loss[-1],elapsed_time))
            
            if( config['save_params'] ):
                print("saving params...")
                torch.save(gen.state_dict(), "./gen_params.pth.tar")
                torch.save(dis.state_dict(), "./dis_params.pth.tar")
                torch.save(dis_optimizer.state_dict(), "./dis_optimizer_state.pth.tar")
                torch.save(gen_optimizer.state_dict(), "./gen_optimizer_state.pth.tar")
                print("saved params.")

#plot errors
utils.save_loss_plot(gen_loss,dis_loss)

#plot generated images
utils.save_result_images(next(iter(dataloader))[0][:15].to(device),generated_imgs[-1],4,config)

#save generated images so see what happened
torch.save(generated_imgs,"gen_imgs_array.pt")

#save gif
utils.save_gif(generated_imgs,4,config)
