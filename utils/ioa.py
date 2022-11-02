import torch 
import numpy as np

def IOA(o, p):
    ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
    return ioa
    focus_ar = torch.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113].cuda()
    
    
    fp = p * focus_ar
    fp[fp==0]=torch.nan
    
    fo = o * focus_ar
    fo[fo==0]=torch.nan
    
    fo = o.detach().cpu().numpy()
    fp = p.detach().cpu().numpy()
    
    o = fo
    s = fp
    id1 = np.where(np.isnan(o))[0]

    s = np.delete(s,id1) # IOA returns nan for blank arrays
    o = np.delete(o,id1)
    
    ioa = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))

    
    return ioa


def val_IOA(o, p):
    full_ioa = 1 -(torch.sum((o-p)**2))/(torch.sum((torch.abs(p-torch.mean(o))+torch.abs(o-torch.mean(o)))**2))
    
    
    focus_ar = torch.zeros((128,174))
    focus_ar[42:70,85:101]=1
    focus_ar=focus_ar[38:70,81:113].cuda()
    
    
    
    
    o = o[0][0]
    p = p[0][0]
    fp = p * focus_ar
    fp[fp==0]=torch.nan
    
    fo = o * focus_ar
    fo[fo==0]=torch.nan
    
    fo = fo.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    
    o = fo.flatten()
    s = fp.flatten()
    id1 = np.where(np.isnan(o))[0]

    s = np.delete(s,id1) # IOA returns nan for blank arrays
    o = np.delete(o,id1)
    
    ioa = 1 -(np.sum((o-s)**2))/(np.sum((np.abs(s-np.mean(o))+np.abs(o-np.mean(o)))**2))
    
    return ioa, full_ioa.detach().cpu().numpy()
