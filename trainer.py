
import time

import gc,os
import datetime
import numpy as np
import pandas as pd

from copy import deepcopy as dc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, random_split

import networks.network
import dataset.dataset

from dataset.custom_dataset import ozone_data

import utils.utils as utils
from utils.ioa import IOA
from utils.norm import normal1, denormal1
from utils.logger import get_logger
from utils.utils import adjust_learning_rate, save_model




def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    logger = get_logger('logs/training_O3_ioa.log')
    
    # cudnn benchmark accelerates the network 
    # (This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.)
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path(opt.save_path)
    utils.check_path(opt.sample_path)

    # Build networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    # trainset = dataset.InpaintDataset(opt)
    
    data_path = "../../O3_inpainting/data/" 
    
    
    focus_ar = np.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113]

    
    camq2016 = np.load(os.path.join(data_path, "O3_CMAQ_2016.npy"))
    camq2017 = np.load(os.path.join(data_path,"O3_CMAQ_2017.npy"))
    camq2018 = np.load(os.path.join(data_path,"O3_CMAQ_2018.npy"))

    cmaq = np.concatenate((camq2016,camq2017,camq2018),axis=0)
    del camq2016,camq2017,camq2018

        
    mask1 = np.load(os.path.join(data_path,"training_mask10.npy"))
    mask2 = np.load(os.path.join(data_path,"training_mask25.npy"))
    mask3 = np.load(os.path.join(data_path,"training_mask40.npy"))
    mask0 = np.load(os.path.join(data_path,"training_mask00.npy"))

    mask = np.concatenate((mask0,mask1,mask2,mask3),axis=0)
    del mask0,mask1,mask2,mask3

    ori = dc(cmaq) #Entire CMAQ data from 2016 to 2018
    ori  = ori[:,38:70,81:113]
    del cmaq
    
    
    np.random.shuffle(mask)
    mask = 1 - mask[:len(ori),38:70,81:113] #Focus Area


    #Normalize the CMAQ data (ppm)
    mx = 0.45
    mn = 0
    ori  = normal1(ori,mn,mx)

    ori =ori#*focus_ar

    gc.collect()

    mask =np.expand_dims(mask, 1)
    data =np.expand_dims(ori, 1)
    
    
    # data, mask = ozone_data(data_path)
    data = torch.Tensor(data)
    mask = torch.Tensor(mask)
    trainset = TensorDataset(data, mask)
       
    # Train_Val Split and Dataloaders
    tv_split = 0.80
    t_split = int(tv_split * len(trainset))
    v_split = int(len(trainset) - t_split)
    train, valid = random_split(trainset,[t_split,v_split])
    train_loader = DataLoader(train, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    valid_loader = DataLoader(valid, batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    print(f"Total number of data: {len(trainset)}")
    print(f"Trainset: {len(train_loader)*opt.batch_size} Validset: {len(valid_loader)*opt.val_batch_size}")
    
    
    # ----------------------------------------
    #            Training and Validation
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    df =pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss', 'Train IOA', 'Val IOA'])
    for epoch in range(opt.epochs):
        
        # Training loop
        for batch_idx, (grayscale, mask) in enumerate(train_loader):
            generator.train()

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 32, 32]
            mask = mask.cuda()                                              # out: [B, 1, 32, 32]
           
            # forward propagation
            optimizer_g.zero_grad()
            out = generator(grayscale, mask)                                # out: [B, 1, 32, 32]
            
            # combining output with image
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]
            
            # Mask L1 Loss
            MaskL1Loss = L1Loss(out_wholeimg, grayscale)
            ioa = IOA(grayscale, out_wholeimg)
            
            # Compute losses
            loss = MaskL1Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [Train IOA : %0.3f] " %
            ((epoch + 1), opt.epochs, batch_idx, len(train_loader), MaskL1Loss.item(), ioa))
            utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1), ioa) 
            exit()
            
        print("Starting Validation Loop")
        # Valudation Loop
        val_loss = 0.0
        val_ioa = 0.0
        prev_val_time = time.time()
        for batch_idx, (val_grayscale, val_mask) in enumerate(valid_loader):
            generator.eval()
            val_grayscale = val_grayscale.cuda()                                    # out: [B, 1, 32, 32]
            val_mask = val_mask.cuda()  

            val_out = generator(val_grayscale, val_mask)                                # out: [B, 1, 32, 32]
            val_out_wholeimg = val_grayscale * (1 - val_mask) + val_out * val_mask 
            
            batch_loss = L1Loss(val_out_wholeimg, val_grayscale).item()
            batch_ioa = IOA(val_grayscale, val_out_wholeimg).item()
            
            print("\r[Epoch %d/%d] [Batch %d/%d] [Val Loss: %.5f] [Val IOA : %0.3f] " %
            ((epoch + 1), opt.epochs, batch_idx, len(valid_loader), batch_loss, batch_ioa))
            
            val_loss += batch_loss
            val_ioa += batch_ioa
            
        val_loss = val_loss/len(valid_loader)   
        val_ioa = val_ioa/len(valid_loader)  
        epochs_rem = opt.epochs - (epoch+1)
        val_time = datetime.timedelta(seconds = epochs_rem * (time.time() - prev_val_time))
        time_left += val_time
        prev_val_time = time.time()
        
        # Print log
        print("\r[Epoch %d/%d]  [Train L1 Loss: %.5f] [Val L1 Loss: %.5f] [Train IOA : %0.3f] [Val IOA : %0.3f] time_left: %s" %
            ((epoch + 1), opt.epochs,  MaskL1Loss.item(), val_loss, ioa, val_ioa,  time_left))
        # logger.info("\r[Epoch %d/%d] [Batch %d/%d] [Train L1 Loss: %.5f] [Val L1 Loss: %.5f] [Train IOA : %0.3f] \
        #       [Val IOA : %0.3f] time_left: %s" %
        #     ((epoch + 1), opt.epochs, batch_idx, len(train_loader), MaskL1Loss.item(), val_loss, ioa, val_ioa, time_left))
        
        ## Saving loss to csv
        dd = [epoch+1, MaskL1Loss.item(), val_loss, ioa, val_ioa,]
        df.loc[len(df)] = dd
        df.to_csv('log.csv')
                   
            

        # Learning rate decrease
        utils.adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        # Save the model
        utils.save_model(generator, (epoch + 1), opt, logger)
        # Save image
        if epoch%opt.img_save_interval == 0:
            utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1))  
        
        
    
       