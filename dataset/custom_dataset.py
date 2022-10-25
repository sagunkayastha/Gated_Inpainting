import numpy as np
import gc,os
from copy import deepcopy as dc
from utils.norm import normal1


def ozone_data(data_path):
        
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
    ori =np.expand_dims(ori, 1)
    
    
    return mask, ori