import torch
import numpy as np
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, device='cuda', logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = [torch.eye(in_channels) * 2**i for i in torch.linspace(0, N_freqs-1, N_freqs)]
        else:
            self.freq_bands = [torch.eye(in_channels) * i for i in torch.linspace(0, N_freqs-1, N_freqs)]
        self.freq_bands = torch.cat(self.freq_bands, dim=-1).to(device)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for func in self.funcs:
            out += [func(x@self.freq_bands)]

        return torch.cat(out, dim=-1)
    
class Gaussian_Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, trainable=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Gaussian_Embedding, self).__init__()
        freq_bands = torch.normal(mean=torch.zeros([in_channels, N_freqs]), std=25.0*torch.ones([in_channels, N_freqs]))
        self.freq_bands = nn.Parameter(freq_bands, trainable)

    def forward(self, x):
        out = [x]
        out += [torch.sin(x@(2.0**self.freq_bands))] 
        out += [torch.cos(x@(2.0**self.freq_bands))]

        return torch.cat(out, dim=-1)

if __name__ == '__main__':
    embed = Gaussian_Embedding(3, 256, 0.1)
    x = torch.randn([100, 3], device='cuda')
    y = embed(x)
    print(y.shape)