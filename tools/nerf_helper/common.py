import torch
from collections import defaultdict
from model.embed import *
from model.nerf import *
from model.siren import *
from model.nerf_diff_act import *
from tools.utils.eval_helper import psnr
from tools.utils.lr_schdule_helper import NeRF_LRDecay
from tools.utils.loss_helper import *
from tools.nerf_helper.render_helper import render_rays

def create_model_family(hparams):
    models = []
    embeddings = []
    if hparams.train.model == 'nerf':
        xyz_encoder = Embedding(in_channels=3, N_freqs=hparams.render.xyz_N_freqs)
        dir_encoder = Embedding(in_channels=3, N_freqs=hparams.render.dir_N_freqs)
        embeddings += [xyz_encoder]
        embeddings += [dir_encoder]
        
        model_corase = NeRF(D=8, W=256, in_channels_xyz=hparams.render.xyz_N_freqs*3*2+3, in_channels_dir=hparams.render.dir_N_freqs*3*2+3)
        model_corase.to('cuda')
        model_fine = NeRF(D=8, W=256, in_channels_xyz=hparams.render.xyz_N_freqs*3*2+3, in_channels_dir=hparams.render.dir_N_freqs*3*2+3)
        model_fine.to('cuda')
        models += [model_corase]
        models += [model_fine]
    elif hparams.train.model == 'siren':
        xyz_encoder = Embedding(in_channels=3, N_freqs=hparams.render.xyz_N_freqs)
        dir_encoder = Embedding(in_channels=3, N_freqs=hparams.render.dir_N_freqs)
        embeddings += [xyz_encoder]
        embeddings += [dir_encoder]
        
        embeddings += [xyz_encoder]
        embeddings += [dir_encoder]
        
        model_corase = Siren_NeRF(D=8, W=256, in_channels_xyz=hparams.render.xyz_N_freqs*3*2+3, in_channels_dir=hparams.render.dir_N_freqs*3*2+3)
        model_corase.to('cuda')
        model_fine = Siren_NeRF(D=8, W=256, in_channels_xyz=hparams.render.xyz_N_freqs*3*2+3, in_channels_dir=hparams.render.dir_N_freqs*3*2+3)
        model_fine.to('cuda')
        models += [model_corase]
        models += [model_fine]
    else:
        raise RuntimeError('Wrong Model Select')
    
    if hparams.train.loss_type == 'mse':
        losser = MSELoss()
    elif hparams.train.loss_type == 'robust_loss':
        # NOTE worse than mse loss
        losser = Adaptive_Loss(num_dims=3, float_dtype=torch.float32, device='cuda:0')
    else:
        raise RuntimeError('Wrong Losser Select!')
    
    grad_var = []
    if hparams.train.loss_type == 'robust_loss':
        grad_var += list(losser.parameters())
    if hparams.train.model == 'gaussian_nerf':
        print('Optim Gaussian Linear!')
        grad_var += list(xyz_encoder.parameters())
        grad_var += list(dir_encoder.parameters())
    for model in models:
        grad_var += list(model.parameters())
        
    optim = torch.optim.Adam(grad_var, lr=hparams.train.lr_init, eps=1e-8, weight_decay=0)
    optim_schdule = NeRF_LRDecay(lr_init=hparams.train.lr_init, lr_final=hparams.train.lr_final,
                                max_steps=hparams.train.num_epochs, lr_delay_steps=hparams.train.lr_delay_steps,
                                lr_delay_mult=hparams.train.lr_delay_mult)
    return models, embeddings, optim, optim_schdule, losser

def train_one_epoch(hparams, before_iter, dataloader, models, embeddings, optim, mse_loss, writer):

    def batch_chunk(rays):
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, hparams.render.chunk):
            result_chunk = render_rays(models=models, embeddings=embeddings, 
                                    rays=rays[i:i+hparams.render.chunk], 
                                    N_samples=hparams.render.N_samples, 
                                    N_importance=hparams.render.N_importance, 
                                    perturb=hparams.render.perturb, 
                                    noise_std=hparams.render.noise_std,
                                    chunk=hparams.render.chunk,
                                    white_back=hparams.render.white_back,
                                    test_time=False)
            for k, v in result_chunk.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results
    
    for cur_iter, samples in enumerate(dataloader):
        rays = samples['rays'].to('cuda')
        rgbs  = samples['rgbs'].to('cuda')
        results = batch_chunk(rays)
        loss = mse_loss(results, rgbs)
        loss.backward()
        optim.step()
        optim.zero_grad()
        with torch.no_grad():
            psnr_value = psnr(results['rgb_fine'], rgbs)
            
        print('Iter: %4.0f' % (before_iter+cur_iter),' Loss: %.10f' % loss.item(), ' PSNR: %.4f' % psnr_value.item())
        
        writer.add_loss(loss.item(), 'train/loss', cur_iter+before_iter)
        writer.add_psnr(psnr_value.item(), 'train/psnr', cur_iter+before_iter)
        
    return cur_iter+1

def batched_inference(hparams, models, embeddings, rays):
    B = rays.shape[0]
    results = defaultdict(list)

    for i in range(0, B, hparams.render.chunk):
        rendered_ray_chunks = render_rays(models=models, embeddings=embeddings, 
                                        rays=rays[i:i+hparams.render.chunk], 
                                        N_samples=hparams.render.N_samples, 
                                        N_importance=hparams.render.N_importance, 
                                        perturb=hparams.render.perturb, 
                                        noise_std=hparams.render.noise_std,
                                        chunk=hparams.render.chunk,
                                        white_back=hparams.render.white_back,
                                        test_time=True)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results