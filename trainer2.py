
import time
import datetime
import os
import pickle
import numpy as np
import cv2

import gc,os
import pandas as pd
import torch
import torch.nn as nn
import itertools
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, random_split


import networks.network
import dataset.dataset
from copy import deepcopy as dc

from dataset.custom_dataset import ozone_data

import utils.utils as utils
from utils.ioa import IOA
from utils.norm import normal1, denormal1
from utils.logger import get_logger


class Trainer:
    
    def __init__(self, opt):
        self.opt = opt
        self.gpu_num = torch.cuda.device_count()
        self.cudnn_benchmark = opt.cudnn_benchmark
        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers 
        
        self.save_path = opt.save_path
        self.sample_path = opt.sample_path
        self.data_path = "../../O3_inpainting/data/" 
        self.logger_path = "logs/"
        self.model_run = "class.log"
        
        self.run()
        
    def prep(self):
        print('Preparing Training')
        data_path = "lat_lon/"
        self.LAT = np.load(os.path.join(data_path,"lat.npy"))
        self.LON = np.load(os.path.join(data_path,"lon.npy"))
        
        self.batch_size *= self.gpu_num
        self.num_workers *= self.gpu_num
        
        utils.check_path(self.save_path)
        utils.check_path(self.sample_path)
        utils.check_path(self.logger_path)
        self.logger = get_logger(os.path.join(self.logger_path, self.model_run))
        self.df = pd.DataFrame(columns=['Epoch', 'Loss', 'IOA'])
        
    def create_models(self):
        print("Preparing Model")
        self.generator = utils.create_generator(self.opt)
        if self.opt.multi_gpu == True:
            self.generator = nn.DataParallel(self.generator)
            self.generator = self.generator.cuda()
        else:
            self.generator = self.generator.cuda()
            
        self.L1Loss = nn.L1Loss()
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr = self.opt.lr_g, betas = (self.opt.b1, self.opt.b2), weight_decay = self.opt.weight_decay)
        
    def IOA(self, o, p):
        
        ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
        return ioa

    # Learning rate decrease
    def adjust_learning_rate(self, optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(self, net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'Gated_Oz_ioa_%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                self.logger.info('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                self.logger.info('The trained model is successfully saved at epoch %d' % (epoch))
    
    def prepare_dataset(self):
        print("Preparing dataset")
        
        data, mask = ozone_data(self.data_path)
        data = torch.Tensor(data)
        mask = torch.Tensor(mask)
        
        trainset = TensorDataset(data, mask) 
        t_split = int(0.80 * len(data))
        v_split = int(len(data) - t_split)
        
        train, valid = random_split(trainset,[t_split,v_split])
        
        
        train_loader = DataLoader(train, batch_size = self.opt.batch_size, shuffle = True, num_workers = self.opt.  num_workers, pin_memory = True)
        valid_loader = DataLoader(valid, batch_size = self.opt.val_batch_size, shuffle = True, num_workers = self.opt.  num_workers, pin_memory = True)
        
        print(f"Total number of data: {len(trainset)}")
        print(len(train_loader)*self.opt.batch_size, len(valid_loader)*self.opt.val_batch_size)
        
        gc.collect()
        
        return train_loader, valid_loader
        
    def summary(self):
        # from torchsummary import summary
                # summary(generator, [grayscale,mask])
        pass
    
    def start_training(self):
        
        # train_loader, valid_loader = self.prepare_dataset()
        prev_time = time.time()
        
        
        data, mask = ozone_data(self.data_path)
        data = torch.Tensor(data)
        mask = torch.Tensor(mask)
        
        trainset = TensorDataset(data, mask) 
        t_split = int(0.80 * len(data))
        v_split = int(len(data) - t_split)
        
        train, valid = random_split(trainset,[t_split,v_split])
        
        
        train_loader = DataLoader(train, batch_size = self.opt.batch_size, shuffle = True, num_workers = self.opt.  num_workers, pin_memory = True)
        valid_loader = DataLoader(valid, batch_size = self.opt.val_batch_size, shuffle = True, num_workers = self.opt.  num_workers, pin_memory = True)
        
        
        
        
        for epoch in range(self.opt.epochs):
            epoch_train_loss = 0.0
            for batch_idx, (grayscale, mask) in enumerate(train_loader):
                self.generator.train()
            # Load and put to cuda
            
                grayscale = grayscale.cuda()                                    # out: [B, 1, 32, 32]
                mask = mask.cuda()                                              # out: [B, 1, 32, 32]
            
            
          
                # forward propagation
                self.optimizer_g.zero_grad()
                out = self.generator(grayscale, mask)                                # out: [B, 1, 32, 32]
                out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

                
                # Mask L1 Loss
                MaskL1Loss = self.L1Loss(out_wholeimg, grayscale)
                o, p = grayscale, out_wholeimg
                ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
                # ioa = self.IOA(grayscale, out_wholeimg)
                print(ioa)
                # Compute losses
                loss = MaskL1Loss
                loss.backward()
                self.optimizer_g.step()

                # Determine approximate time left
                batches_done = epoch * len(train_loader) + batch_idx
                batches_left = self.opt.epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
                prev_time = time.time()

                print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [IOA : %0.3f] time_left: %s" %
                    ((epoch + 1), self.opt.epochs, batch_idx, len(train_loader), MaskL1Loss.item(), ioa, time_left))
                self.logger.info("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [IOA : %0.3f] time_left: %s" %
                    ((epoch + 1), self.opt.epochs, batch_idx, len(train_loader), MaskL1Loss.item(), ioa, time_left))
                
                utils.sample(grayscale, mask, out_wholeimg, self.opt.sample_path, (epoch + 1), self.LAT, self.LON)
                exit()
                
            print("Running Validation Loop")
                # Validation loop
            val_loss = 0.0
            val_ioa = 0.0
            for batch_idx, (val_grayscale, val_mask) in enumerate(valid_loader):
                self.generator.eval()
                val_grayscale = val_grayscale.cuda()                                    # out: [B, 1, 32, 32]
                val_mask = val_mask.cuda()  

                val_out = self.generator(val_grayscale, val_mask)                                # out: [B, 1, 32, 32]
                val_out_wholeimg = val_grayscale * (1 - val_mask) + val_out * val_mask 
                
                val_loss += self.L1Loss(val_out_wholeimg, val_grayscale)
                val_ioa += IOA(val_grayscale, val_out_wholeimg)
            
                # Print log
                
                
            print(f"val loss = {val_loss/len(valid_loader)}")
            exit()
            self.df.to_csv('log.csv')
            
            if batch_idx%50 == 0:
                dd = [epoch+1, MaskL1Loss.item(), ioa]
                self.df.loc[len(self.df)] = dd
        
        
    
    
    def run(self):
        self.prep()
        self.create_models()
        self.start_training()
    
    

# def Trainer2(opt):
#     # ----------------------------------------
#     #      Initialize training parameters
#     # ----------------------------------------
#     logger = get_logger('logs/training_O3_ioa.log')
#     plot_stuff = {'Epoch':[], 
#                   'Loss':[], 
#                   'IOA': []}
#     # cudnn benchmark accelerates the network
#     cudnn.benchmark = opt.cudnn_benchmark

#     # Handle multiple GPUs
#     gpu_num = torch.cuda.device_count()
#     print("There are %d GPUs used" % gpu_num)
#     opt.batch_size *= gpu_num
#     opt.num_workers *= gpu_num
#     print("Batch size is changed to %d" % opt.batch_size)
#     print("Number of workers is changed to %d" % opt.num_workers)
    
#     # Build path folder
#     utils.check_path(opt.save_path)
#     utils.check_path(opt.sample_path)

#     # Build networks
#     generator = utils.create_generator(opt)

#     # To device
#     if opt.multi_gpu == True:
#         generator = nn.DataParallel(generator)
#         generator = generator.cuda()
#     else:
#         generator = generator.cuda()

#     # Loss functions
#     L1Loss = nn.L1Loss()

#     # Optimizers
#     optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    
    
    
#     # ----------------------------------------
#     #       Initialize training dataset
#     # ----------------------------------------

#     # Define the dataset
#     # trainset = dataset.InpaintDataset(opt)
    
    


    
    
    
#     print('The overall number of images equals to %d' % len(trainset))
#     logger.info('The overall number of images equals to %d' % len(trainset))
    
#     # Define the dataloader
#     dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
#     # ----------------------------------------
#     #            Training and Testing
#     # ----------------------------------------

#     # Initialize start time
#     prev_time = time.time()
#     import matplotlib.pyplot as plt
#     # Training loop
#     df =pd.DataFrame(columns=['Epoch', 'Loss', 'IOA'])
#     for epoch in range(opt.epochs):
#         for batch_idx, (grayscale, mask) in enumerate(dataloader):

#             # Load and put to cuda
#             grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
#             mask = mask.cuda()                                              # out: [B, 1, 256, 256]
           
#             # from torchsummary import summary
#             # summary(generator, [grayscale,mask])
            
#             # exit()
#             # forward propagation
#             optimizer_g.zero_grad()
#             out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            
            
            
#             out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

#             # Mask L1 Loss
#             MaskL1Loss = L1Loss(out_wholeimg, grayscale)
#             ioa = IOA(grayscale, out_wholeimg)
#             # Compute losses
#             loss = MaskL1Loss
#             loss.backward()
#             optimizer_g.step()

#             # Determine approximate time left
#             batches_done = epoch * len(dataloader) + batch_idx
#             batches_left = opt.epochs * len(dataloader) - batches_done
#             time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
#             prev_time = time.time()

#             # Print log
#             print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [IOA : %0.3f] time_left: %s" %
#                 ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), ioa, time_left))
#             logger.info("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [IOA : %0.3f] time_left: %s" %
#                 ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), ioa, time_left))
            
           
#             df.to_csv('log.csv')
            
#             if batch_idx%10 == 0:
#                 dd = [epoch+1, MaskL1Loss.item(), ioa]
#                 df.loc[len(df)] = dd
            
            
                
                
#         # Learning rate decrease
#         adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        
#         # Save the model
#         save_model(generator, (epoch + 1), opt)
#         utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1))
        

