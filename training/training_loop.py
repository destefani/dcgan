import os
import time

import numpy as np
import torch
import torchvision.datasets as datasets




















def training_loop(
    dataroot,
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
    ngpu=1
):
    dataset = dataset



    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):


