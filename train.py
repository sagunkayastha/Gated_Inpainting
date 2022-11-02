
import os
from configuration import parse

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    
    opt = parse()
    
    # '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # '''
    
    # Enter main function
    from trainer import Trainer
    
    obj = Trainer(opt)
        
    