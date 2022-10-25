import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
import random
import networks.network as network
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from utils.plot_domain import plot_domain, plot_domain_s, plot_domain_val
from utils.norm import denormal1
# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GrayInpaintingNet(opt)
    print('Generator is created!')
    # Init the networks
    
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % opt.finetune_path)
    else:
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
        
        
    
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Pre-trained VGG-16
    vgg16 = torch.load('vgg16_pretrained.pth')
    # Get the first 16 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Update the parameters
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet

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
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def focus_IOA(o, p):
    
    focus_ar = np.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113]
    
    o = o
    p = p
    
    fp = p * focus_ar
    fp[fp==0]=np.nan
    
    
    
    
    fo = o * focus_ar
    fo[fo==0]=np.nan
    
    
    o = fo.flatten()
    s = fp.flatten()
    id1 = np.where(np.isnan(o))[0]

    s = np.delete(s,id1) # IOA returns nan for blank arrays
    o = np.delete(o,id1)
    
    ioa = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    
    return ioa

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def sample(grayscale, mask, out, save_folder, file_name, ioa):
    # to cpu
    grayscale = grayscale[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    mask = mask[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    out = out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    
   
    
    grayscale = denormal1(grayscale, 0, 0.45)
    out = denormal1(out, 0, 0.45)
    
    masked_img = grayscale * (1 - mask) + mask
    
    
    
    focus_ioa = focus_IOA(grayscale, out)
    
    # out = out * focus_ar
    # out[out==0]=np.nan
    
    
    data_path = "lat_lon/"
    fig = plot_domain_val(data_path,  grayscale,  masked_img, out, ioa, focus_ioa)
    imgname = os.path.join(save_folder, 'epoch_'+str(file_name)+'.png')
    plt.savefig(imgname, dpi=200)
    
    
def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

def create_mask(mask):
    
    mask[np.isnan(mask)]=False
    
    mask[mask>0]=1
    percent_missing = [65,75,80] 
    mask[mask==0]=np.nan
    percent = random.choice(percent_missing)
    
    
    # exit()
    hr_mask = dc(mask)
    
    # Get a vector of 1-d indexed indexes of non NaN elements
    indices = np.where(np.isfinite(hr_mask).ravel())[0]
    
    # Shuffle the indices, specify percentage of stations to be removed (rounded down with int())
    to_replace = np.random.permutation(indices)[:int(indices.size*(percent/100))]
    
    # Replace those indices with 0 (ignoring NaNs)
    hr_mask[np.unravel_index(to_replace, hr_mask.shape)] = 0
    hr_mask = hr_mask[:,:,np.newaxis] #Add new axis

    hr_mask[np.isnan(hr_mask)]=False
    
    
    
    return 1-hr_mask

def sample_val(grayscale, mask, out, save_folder, file_name, ioa, full_ioa):
    # to cpu
    grayscale = grayscale[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    mask = mask[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    out = out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    focus_ar = np.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113]
    
    grayscale = denormal1(grayscale, 0, 0.45)
    out = denormal1(out, 0, 0.45)
    
    masked_img = grayscale * (1 - mask) + mask

    # out = out * focus_ar
    # out[out==0]=np.nan
    
    
    data_path = "lat_lon/"
    fig = plot_domain_val(data_path,  grayscale,  masked_img, out, ioa, full_ioa)
    imgname = os.path.join(save_folder, file_name)
    plt.savefig(imgname)
    
    
# Learning rate decrease
def adjust_learning_rate(optimizer, epoch, opt, init_lr):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Save the model if pre_train == True
def save_model(net, epoch, opt, logger):
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = 'Gated_Oz_ioa_%d_batchsize%d.pth' % (epoch, opt.batch_size)
    model_path = os.path.join(opt.save_path, model_name)
    if opt.multi_gpu == True:
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.module.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))
            logger.info('The trained model is successfully saved at epoch %d' % (epoch))
    else:
        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))
            logger.info('The trained model is successfully saved at epoch %d' % (epoch))