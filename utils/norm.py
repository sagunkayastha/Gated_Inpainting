
import numpy as np
import torch


def normal1(v,min_v,max_v):
    min_mat = min_v * (np.ones(np.shape(v)))
    x = (v - min_mat)/(max_v-min_v)
    x = (x *0.8) + 0.1
    return x

def denormal1(v,min_v,max_v):
    min_mat = min_v * (np.ones(np.shape(v)))
    x =( ((v-0.1)/0.8) * (max_v-min_v)) + min_mat
    return x


# def denormal1(v,min_v,max_v):
#     min_mat = min_v * (torch.ones(v.shape).cuda())
    
#     print(type(min_mat), type(min_v), type(max_v))
#     x =( ((v-0.1)/0.8) * (max_v-min_v)) + min_mat
    # return x