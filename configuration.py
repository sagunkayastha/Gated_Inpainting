import configparser
class obj(object):
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(k, (list, tuple)):
                    setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
                else:
                    setattr(self, k, obj(v) if isinstance(v, dict) else v)

def parse():
    config = configparser.ConfigParser()
    config.read('system_config.ini')
    
    CONFIG = {}
    CONFIG['run_name'] = config.get('MODEL', 'run_name')
    CONFIG['pre_train'] = config.getboolean('MODEL', 'pre_train')
    CONFIG['finetune_path'] = config.get('MODEL', 'finetune_path')
    CONFIG['in_channels'] = config.getint('MODEL', 'in_channels')
    CONFIG['mask_channels'] = config.getint('MODEL', 'mask_channels')
    CONFIG['latent_channels'] = config.getint('MODEL', 'latent_channels')
    CONFIG['out_channels'] = config.getint('MODEL', 'out_channels')
    CONFIG['pad'] = config.get('MODEL', 'pad')
    CONFIG['activ_g'] = config.get('MODEL', 'activ_g')
    CONFIG['norm_g'] = config.get('MODEL', 'norm_g')
    CONFIG['init_type'] = config.get('MODEL', 'init_type')
    CONFIG['init_gain'] = config.getfloat('MODEL', 'init_gain')
    
    CONFIG['baseroot'] = config.get('PATH', 'baseroot')
    CONFIG['save_path'] = config.get('PATH', 'save_path')
    CONFIG['sample_path'] = config.get('PATH', 'sample_path')
    CONFIG['data_path'] = config.get('PATH', 'data_path')
    CONFIG['logger_path'] = config.get('PATH', 'logger_path')
    CONFIG['test_path'] = config.get('PATH', 'test_path')
    CONFIG['finetune_path'] = config.get('PATH', 'finetune_path')
    CONFIG['test_model_path'] = config.get('PATH', 'test_model_path')
    
    CONFIG['multi_gpu'] = config.getboolean('TRAINING', 'multi_gpu')
    CONFIG['gpu_ids'] = config.get('TRAINING', 'gpu_ids')
    CONFIG['cudnn_benchmark'] = config.getboolean('TRAINING', 'cudnn_benchmark')
    CONFIG['epochs'] = config.getint('TRAINING', 'epochs')
    CONFIG['checkpoint_interval'] = config.getint('TRAINING', 'checkpoint_interval')
    CONFIG['img_save_interval'] = config.getint('TRAINING', 'img_save_interval')
    CONFIG['train_batch_size'] = config.getint('TRAINING', 'train_batch_size')
    CONFIG['val_batch_size'] = config.getint('TRAINING', 'val_batch_size')
    CONFIG['imgsize'] = config.getint('TRAINING', 'imgsize')
    CONFIG['lr_g'] = config.getfloat('TRAINING', 'lr_g')
    CONFIG['b1'] = config.getfloat('TRAINING', 'b1')
    CONFIG['b2'] = config.getfloat('TRAINING', 'b2')
    CONFIG['weight_decay'] = config.getfloat('TRAINING', 'weight_decay')
    CONFIG['lr_decrease_epoch'] = config.getfloat('TRAINING', 'lr_decrease_epoch')
    CONFIG['lr_decrease_factor'] = config.getfloat('TRAINING', 'lr_decrease_factor')
    CONFIG['num_workers'] = config.getint('TRAINING', 'num_workers')
    
    CONFIG['test_whole'] = config.getboolean('TEST', 'test_whole')
    CONFIG['test_samples'] = config.getint('TEST', 'test_samples')
    
    return obj(CONFIG)
    
if __name__ == '__main__':
    config = parse()
                
    print(config.lr_g)