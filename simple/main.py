import time

import gc,os
import datetime
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from copy import deepcopy as dc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, random_split

# from dataset.custom_dataset import ozone_data
import network

import matplotlib.pyplot as plt


## Variables
multi_gpu = True
gpu_ids = 0,1
cudnn_benchmark = True
num_workers = 0
data_path = "/dataFs/skayasth/2022/Sept/O3_inpainting/data/"
save_path = "models"
sample_path = "samples"
finetune_path = ""
test_model_path = ""
img_save_interval = 10
checkpoint_interval = 10

## Run 
run_name = "test2"
train_batch_size = 128
val_batch_size = 128
epochs = 50
imgsize = 32
lr_g = 2e-4
b1 = 0.5
b2 = 0.999
weight_decay = 0
lr_decrease_epoch = 25
lr_decrease_factor = 0.5

# Model Variables
in_channels = 1
mask_channels = 1 
latent_channels = 64
out_channels = 1
pad = "reflect"
activ_g = "lrelu"
norm_g = "bn" 
init_type = "normal"
init_gain = 0.02




def IOA(o, p):
    ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
    return ioa

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Helper (Convert dictionary to object)
class obj(object):
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(k, (list, tuple)):
                    setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
                else:
                    setattr(self, k, obj(v) if isinstance(v, dict) else v)

def normal1(v,min_v,max_v):
    min_mat = min_v * (np.ones(np.shape(v)))
    x = (v - min_mat)/(max_v-min_v)
    x = (x *0.8) + 0.1
    return x

def denormal1(v,min_v,max_v):
    min_mat = min_v * (np.ones(np.shape(v)))
    x =( ((v-0.1)/0.8) * (max_v-min_v)) + min_mat
    return x



def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def create_generator(opt, test=False):
    # Initialize the networks
    
    
    generator = network.GrayInpaintingNet(opt)
    print('Generator is created!')
    # Init the networks
    
    if finetune_path:
        pretrained_net = torch.load(finetune_path)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % finetune_path)
        
    elif test:
        pretrained_net = torch.load(test_model_path)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % test_model_path)
        
    else:
        network.weights_init(generator, init_type = init_type, init_gain = init_gain)
        print('Initialize generator with %s type' % init_type)
        

    return generator

   
# Learning rate decrease
def adjust_learning_rate(optimizer, epoch, opt, init_lr):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Save the model if pre_train == True
def save_model(net, epoch ):
    
    global train_batch_size, save_path, checkpoint_interval, multi_gpu
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = 'Gated_Oz_ioa_%d_batchsize%d.pth' % (epoch, train_batch_size)
    model_path = os.path.join(save_path, model_name)
    if multi_gpu == True:
        if epoch % checkpoint_interval == 0:
            torch.save(net.module.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))
            
    else:
        if epoch % checkpoint_interval == 0:
            torch.save(net.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))



def plot_domain(lat_path, arr1, arr2, arr3, ioa):
    
    
    if len(arr1.shape) == 3:
        arr1 = arr1[:,:,0]
    if len(arr2.shape) == 3:
        arr2 = arr2[:,:,0]
    
    if len(arr3.shape) == 3:
        arr3 = arr3[:,:,0]
    
    LAT = np.load(os.path.join(lat_path,"lat.npy"))
    LON = np.load(os.path.join(lat_path,"lon.npy"))

    lat1 = LAT[38:70,81:113]
    lon1 = LON[38:70,81:113]
    fig = plt.figure(figsize=(15, 10)) #Define size of figure
    title = "Imputed" 
    # fig.suptitle(title,y=0.95,fontsize=20,fontweight="bold")
    cmap = plt.get_cmap('Set2', 3)

    # ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())

    ax.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax.set_title('Batch_real', fontsize=15)
    cm=ax.pcolormesh(lon1,lat1,arr1,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax, fraction=0.046, pad=0.04) 
    
    
    arr2[arr2==1]=np.nan
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax2.set_title('Masked Input', fontsize=15)
    cm=ax2.pcolormesh(lon1,lat1,arr2,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax2.coastlines(resolution='10m')
    ax2.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax2, fraction=0.046, pad=0.04) 
    
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax3.set_title('Imputed', fontsize=15)
    cm=ax3.pcolormesh(lon1,lat1,arr3,cmap="YlOrRd",vmin=0, vmax=arr1.max(), transform=ccrs.PlateCarree())
    ax3.coastlines(resolution='10m')
    ax3.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax3, fraction=0.046, pad=0.04) 
    
    ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.set_extent([125.5, 132.5, 33, 40],crs=ccrs.PlateCarree())
    ax4.set_title('Bias P-O', fontsize=15)
    cm=ax4.pcolormesh(lon1,lat1,arr1-arr3,cmap='Reds_r',vmin=(arr3 - arr1).min(), vmax=(arr1-arr3).max(), transform=ccrs.PlateCarree())
    ax4.coastlines(resolution='10m')
    ax4.add_feature(cfeature.BORDERS)
    fig.colorbar(cm, ax=ax4, fraction=0.046, pad=0.04) 
    

    

    fig.suptitle(f"IOA : {ioa:.3f}") 
    
    return fig

def sample(data_path, grayscale, mask, out, save_folder, file_name, ioa):
    # to cpu
    grayscale = grayscale[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    mask = mask[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    out = out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    grayscale = denormal1(grayscale, 0, 0.45)
    out = denormal1(out, 0, 0.45)
    
    masked_img = grayscale * (1 - mask) + mask
    # data_path = "lat_lon/"
    fig = plot_domain(data_path,  grayscale,  masked_img, out, ioa)
    imgname = os.path.join(save_folder, 'epoch_'+str(file_name)+'.png')
    plt.savefig(imgname, dpi=200)



### MAIN FUNCTION
def main():
    
    global cudnn_benchmark , train_batch_size, num_workers, \
        data_path, save_path, sample_path, finetune_path, test_model_path
    
    # cudnn benchmark accelerates the network 
    # (This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.)
    cudnn.benchmark = cudnn_benchmark
    
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    train_batch_size *= gpu_num
    num_workers *= gpu_num
    print("Batch size is changed to %d" % train_batch_size)
    print("Number of workers is changed to %d" % num_workers)
    
    
     # Build base path folder    
    check_path(save_path)
    check_path(sample_path)

    # update to model_specific path
    save_path = os.path.join(save_path, run_name)
    sample_path = os.path.join(sample_path, run_name)
    check_path(save_path)
    check_path(sample_path)
    
    
    opt = {"in_channels": in_channels,
           "latent_channels": latent_channels,
           "mask_channels": mask_channels,
           "activ_g": activ_g,
           "norm_g": norm_g,
           "pad": pad,
           "out_channels": out_channels,
           "lr_decrease_factor": lr_decrease_factor,
           "lr_decrease_epoch": lr_decrease_epoch,
           "multi_gpu": multi_gpu,
           "train_batch_size": train_batch_size}
    opt = obj(opt)
    
    generator = create_generator(opt)
    
    if multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()    
    
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = lr_g, betas = (b1, b2), weight_decay = weight_decay)
    
    
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    
    
    focus_ar = np.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113]

    
    cmaq2016 = np.load(os.path.join(data_path, "O3_CMAQ_2016.npy"))
    # cmaq2017 = np.load(os.path.join(data_path,"O3_CMAQ_2017.npy"))
    # camq2018 = np.load(os.path.join(data_path,"O3_CMAQ_2018.npy"))

    # cmaq = np.concatenate((cmaq2016,cmaq2017,cmaq2018),axis=0)
    # del camq2016,camq2017,camq2018
    cmaq = cmaq2016

        
    mask1 = np.load(os.path.join(data_path,"training_mask10.npy"))
    # mask2 = np.load(os.path.join(data_path,"training_mask25.npy"))
    # mask3 = np.load(os.path.join(data_path,"training_mask40.npy"))
    # mask0 = np.load(os.path.join(data_path,"training_mask00.npy"))

    # mask = np.concatenate((mask0,mask1,mask2,mask3),axis=0)
    # del mask0,mask1,mask2,mask3
    mask = mask1
    
    
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
    train_loader = DataLoader(train, batch_size = train_batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    valid_loader = DataLoader(valid, batch_size = val_batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)
    print(f"Total number of data: {len(trainset)}")
    print(f"Trainset: {len(train_loader)*train_batch_size} Validset: {len(valid_loader)*val_batch_size}")
    
    prev_time = time.time()
    
    df =pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss', 'Train IOA', 'Val IOA'])
    
    for epoch in range(epochs):
        
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
            batches_left = epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [Train IOA : %0.3f] " %
            ((epoch + 1), epochs, batch_idx, len(train_loader), MaskL1Loss.item(), ioa))
            
            
            
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
            ((epoch + 1), epochs, batch_idx, len(valid_loader), batch_loss, batch_ioa))
            
            val_loss += batch_loss
            val_ioa += batch_ioa
            
            
        val_loss = val_loss/len(valid_loader)   
        val_ioa = val_ioa/len(valid_loader)  
        epochs_rem = epochs - (epoch+1)
        val_time = datetime.timedelta(seconds = epochs_rem * (time.time() - prev_val_time))
        time_left += val_time
        prev_val_time = time.time()
        
        # Print log
        print("\r[Epoch %d/%d]  [Train L1 Loss: %.5f] [Val L1 Loss: %.5f] [Train IOA : %0.3f] [Val IOA : %0.3f] time_left: %s" %
            ((epoch + 1), epochs,  MaskL1Loss.item(), val_loss, ioa, val_ioa,  time_left))
        # logger.info("\r[Epoch %d/%d] [Batch %d/%d] [Train L1 Loss: %.5f] [Val L1 Loss: %.5f] [Train IOA : %0.3f] \
        #       [Val IOA : %0.3f] time_left: %s" %
        #     ((epoch + 1), opt.epochs, batch_idx, len(train_loader), MaskL1Loss.item(), val_loss, ioa, val_ioa, time_left))
        
        ## Saving loss to csv
        
        dd = [epoch+1, MaskL1Loss.item(), val_loss, ioa.item(), val_ioa,]
        
        df.loc[len(df)] = dd
        log_filename = os.path.join(run_name + '.csv')
        df.to_csv(log_filename)
                   
            

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, lr_g)
        # Save the model
        save_model(generator, (epoch + 1))
        # Save image
        if epoch%img_save_interval == 0:
            sample(data_path, grayscale, mask, out_wholeimg, sample_path, (epoch + 1), ioa) 
        
    
    
    
    
if __name__ == "__main__":
    main()