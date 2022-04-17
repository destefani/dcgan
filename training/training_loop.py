import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from networks import Generator, Discriminator

#----------------------------------------------------------------------------

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

#----------------------------------------------------------------------------

def training_loop(
    dataroot,
    outdir,
    num_epochs,
    workers,
    batch_size,
    image_size,
    num_channels,
    z_dim,
    gen_features_size,
    disc_features_size,
    lr,
    beta1,
    ngpu,
    loss=False,  # to-do

):

    # Load data
    dataset = dataset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Initialize models
    generator = Generator(
        z_dim,
        num_channels,
        gen_features_size,
        ngpu).to(device)
    generator.apply(weights_init)

    
    discriminator = Discriminator(
        num_channels,
        disc_features_size,
        ngpu).to(device)
    discriminator.apply(weights_init)

    # Initialize optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(image_size, z_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # 1. Update discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with all-real batch
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            # Train with all-fake batch
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            # 1. Update generator: maximize log(D(G(z)))
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1