import os
import torch
from tensorboardX import SummaryWriter

class LOG(object):
    def __init__(self, exp_name) -> None:
        self.exp_name = exp_name
        self.exp_path = os.path.join('temp/exp', self.exp_name)
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)
        self.logger = SummaryWriter(self.exp_path)
    
    def add_loss(self, loss, name, step):
        self.logger.add_scalar(name, loss, step)
    
    def add_image(self, img, name, step):
        self.logger.add_image(name, img, step, dataformats='HWC')
    
    def add_psnr(self, psnr, name, step):
        self.logger.add_scalar(name, psnr, step)
        
    def save_ckpts(self, model, name):
        save_path = os.path.join(self.exp_path, 'ckpts')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ret = {}
        ret['corase'] = model[0]
        ret['fine'] = model[1]
        torch.save(ret, os.path.join(save_path, name))