
import random
import PIL

import torch
from ProjectorEnvironment import ProjectorEnvironment
from cGAN import Discriminator, Generator
import torchvision
from torch import nn

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create the models
    gmodel = Generator(3,3)
    dmodel = Discriminator()
    
    # Load the weights
    
    # Create the projector environment
    env = ProjectorEnvironment("balloons.png")
    disc_optimizer = torch.optim.Adam(dmodel.parameters(), lr=0.001)
    gen_optimizer = torch.optim.Adam(gmodel.parameters(), lr=0.001)
    # for item in epoch
    for i in range(500):
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        # Get labeled image
        label_image = torchvision.io.read_image("balloons/label.png")
        # Load the image and configuration to be projected and take a snapshot
        env.newRandomWall()
        env.Snapshot("out.png")
        # Run image through Generator
        proj_image = torchvision.io.read_image("balloons/out.png").type(torch.float32).reshape(1,3,256,256).to(device)
        processed_image = gmodel(proj_image)
        # convert tensor to image to save
        processed_image = processed_image.type(torch.uint8).reshape(3,256,256).detach()
        torchvision.io.write_png(processed_image, "balloons/processed.png")
        # Set new processed image in projector and get snapshot
        env.NewImage("balloons/processed.png")
        env.Snapshot("processed_projected.png")
        proc_proj_image = torchvision.io.read_image("balloons/processed_projected.png").type(torch.float32).reshape(1,3,256,256).to(device)
        disc_real_label = torch.ones((1), device=device)
        proc_proj_image = torchvision.io.read_image("balloons/out.png").type(torch.float32).reshape(1,3,256,256)
        disc_gen_label = torch.zeros((1), device=device)
        # Run generated image and original through Discriminator
        disc_gen_output = dmodel(proc_proj_image).view(-1)
        disc_org_output = dmodel(proj_image).view(-1)
        # Update Discrminator weights
        disc_gen_loss = nn.BCELoss()(disc_gen_output, disc_gen_label)
        disc_org_loss = nn.BCELoss()(disc_org_output, disc_real_label)
        disc_total_loss = disc_gen_loss + disc_org_loss
        disc_total_loss.backward()
        disc_optimizer.step()
        # Update Generator weights
        gen_l1_loss = nn.L1Loss()(label_image, proc_proj_image)
        geb_dic_output = dmodel(proc_proj_image).view(-1)
        gen_disc_gen_loss = nn.BCELoss()(geb_dic_output, disc_gen_label)
        gen_total_loss = gen_disc_gen_loss + (100 * gen_l1_loss)
        gen_total_loss.backward()
        gen_optimizer.step()
        # Log
        print(f"Epoch: {i}, Generator Loss: {gen_total_loss}, Discriminator Loss: {gen_total_loss}")
       
if __name__ == "__main__":
    train()