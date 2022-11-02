
from re import I
import gc
import os
import numpy as np
import pandas as pd
from configuration import parse
from copy import deepcopy as dc

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import utils.utils as utils
from utils.ioa import IOA, val_IOA
from utils.norm import normal1, denormal1


def Test(opt):
    # Build base path folder
    utils.check_path(opt.test_path)
    generator = utils.create_generator(opt, test=True).cuda()

    focus_ar = np.zeros((128, 174))
    focus_ar[42:70, 85:101] = 1
    focus_ar = focus_ar[38:70, 81:113]

    # Load Test Data
    cmaq = np.load(os.path.join(opt.data_path, "O3_CMAQ_2019.npy"))
    mask = np.load(os.path.join(opt.data_path, "training_mask40.npy"))

    ori = dc(cmaq)  # Entire CMAQ data from 2016 to 2018
    ori = ori[:, 38:70, 81:113]
    del (cmaq)

    mx = 0.45
    mn = 0
    ori = normal1(ori, mn, mx)

    mask = 1 - mask[:len(ori), 38:70, 81:113]

    mask = np.expand_dims(mask, 1)
    ori = np.expand_dims(ori, 1)

    gc.collect()
    ori = torch.Tensor(ori)
    mask = torch.Tensor(mask)

    # Forward
    val_set = TensorDataset(ori, mask)

    # Full test_set
    # Valudation Loop
    if opt.test_whole == True:
        dataloader = DataLoader(
            val_set, batch_size=16, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        test_ioa = 0.0
        for batch_idx, (test_img, test_mask) in enumerate(dataloader):
            generator.eval()
            # out: [B, 1, 32, 32]
            test_img = test_img.cuda()
            test_mask = test_mask.cuda()

            # out: [B, 1, 32, 32]
            val_out = generator(test_img, test_mask)
            val_out_wholeimg = test_img * (1 - test_mask) + val_out * test_mask

            batch_ioa = IOA(test_img, val_out_wholeimg).item()

            print(" [Batch %d/%d]  [Val IOA : %0.3f] " %
            ((batch_idx, len(dataloader),  batch_ioa)))

            test_ioa += batch_ioa

        val_ioa=val_ioa/len(valid_loader)

        # Final IOA
        print("Final Ioa = {val_ioa}")

    else:
        dataloader = DataLoader(
            val_set, batch_size=16, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        for batch_idx, (grayscale, mask) in enumerate(dataloader):

            # out: [B, 1, 32, 32]
            grayscale=grayscale.cuda()
            mask=mask.cuda()

            with torch.no_grad():
                # out: [B, 1, 32, 32]
                out=generator(grayscale, mask)
                out_wholeimg=grayscale * (1 - mask) + out * mask

                ioa, full_ioa=val_IOA(grayscale.clone(), out_wholeimg.clone())
        
                # ioa, full_ioa = 0 ,0
                img_name=str(batch_idx) + '.png'
                utils.sample_val(opt.data_path, grayscale, mask, out,
                                 opt.test_path, img_name, ioa, full_ioa)

                if batch_idx == opt.test_samples:
                    break


if __name__ == "__main__":
    opt=parse()
    obj=Test(opt)
