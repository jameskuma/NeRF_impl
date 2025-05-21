from torch.utils.data import DataLoader
from data.blender_dataset import BlenderDataset
from data.llff_dataset import LLFFDataset
from tools.utils.eval_helper import psnr
from tools.nerf_helper.common import create_model_family, train_one_epoch, batched_inference
from tools.utils.option_helper import get_opts, load_config
from tools.utils.logger import LOG

import os
import imageio
import torch
import random
import numpy as np
from rich.progress import track

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    setup_seed(66)
    
    args = get_opts()
    exp_ver = args.exp_ver
    hparams = load_config(args.config_dir, './config/base.yaml')
    hparams.exp.exp_name = hparams.exp.exp_name + exp_ver
    log = LOG(exp_name=hparams.exp.exp_name)

    if hparams.dataset.dataset_name == 'llff':
        dataset = LLFFDataset(root_dir=hparams.dataset.root_dir, split='train', img_wh=(504, 378))
        val = LLFFDataset(root_dir=hparams.dataset.root_dir, split='val', img_wh=(504, 378))
    elif hparams.dataset.dataset_name == 'lego':
        dataset = BlenderDataset(root_dir=hparams.dataset.root_dir, split='train', img_wh=(400, 400))
        val = BlenderDataset(root_dir=hparams.dataset.root_dir, split='val', img_wh=(400, 400))
    else:
        raise RuntimeError('Wrong Dataset Select!')

    dataloader = DataLoader(dataset, shuffle=True, batch_size=hparams.train.batch_size)
    models, embeddings, optim, optim_schdule, losser = create_model_family(hparams)
    
    before_iter = 0
    val_step = 0
    
    for epoch in track(range(hparams.train.num_epochs)):
        cur_iter = train_one_epoch(hparams, before_iter, dataloader, 
                                models, embeddings, optim, losser, log)
        before_iter += cur_iter
        optim_schdule.step(optim, epoch)

        if (epoch%5 == 0) or (epoch == hparams.train.num_epochs-1) or epoch==0:
            imgs = []
            img_gts = []
            psnr_value = 0.0
            with torch.no_grad():
                for i in range(len(val)):
                    sample = val[0]
                    rays = sample['rays'].cuda()
                    rgbs = sample['rgbs'].cuda()
                    results = batched_inference(hparams, models, embeddings, rays)
                
                    psnr_value += psnr(results['rgb_fine'], rgbs).item()
                
                    img_pred = results['rgb_fine'].view(hparams.dataset.img_wh[1], hparams.dataset.img_wh[0], 3).cpu().numpy()
                    img_gt = sample['rgbs'].view(hparams.dataset.img_wh[1], hparams.dataset.img_wh[0], 3).cpu().numpy()
                    img_pred_ = (np.clip(img_pred, 0, 1)*255).astype(np.uint8)
                    img_gt = (np.clip(img_gt, 0, 1)*255).astype(np.uint8)
                    
                    log.add_image(img_pred_, 'val/img_pred', val_step)
                    log.add_image(img_gt, 'val/img_gt', val_step)
                    val_step += 1
                    imgs += [img_pred_]

                psnr_value /= len(val)
                log.add_psnr(psnr_value, 'val/psnr', val_step)
    dir_name = './temp/img'
    imageio.mimsave(os.path.join(dir_name, hparams.dataset.dataset_name + '.gif'), imgs)
    log.save_ckpts(models, hparams.dataset.dataset_name + '_' + exp_ver + '.tar')