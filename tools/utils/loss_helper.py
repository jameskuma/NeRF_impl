from torch import nn
import robust_loss_pytorch
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss

class Smooth_L1_Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(Smooth_L1_Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean', beta=beta)

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss

class Adaptive_Loss(nn.Module):
    def __init__(self, num_dims, float_dtype=torch.float32, device='cpu'):
        super().__init__()
        self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims, float_dtype, device)

    def forward(self, inputs, targets):
        error = inputs['rgb_coarse']-targets
        loss = self.adaptive.lossfun(error).sum(-1)
        loss = torch.mean(loss, dim=-1)
        if 'rgb_fine' in inputs:
            error = inputs['rgb_fine']-targets
            loss_fine = self.adaptive.lossfun(error).sum(-1)
            loss_fine = torch.mean(loss_fine, dim=-1)
            loss += loss_fine
        return loss