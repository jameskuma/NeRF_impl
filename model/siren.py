import torch
from torch import nn
from siren_pytorch import Siren

# NOTE Rewrite Linear Layer using Initalization According to Activation
class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

class Siren_NeRF(nn.Module):
    def __init__(self,
                D=8, W=256,
                in_channels_xyz=63, in_channels_dir=27):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        """
        super(Siren_NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        # xyz encoding layers
        self.layer0 = nn.Sequential(
            Siren(in_channels_xyz, W, w0=30., is_first=True), 
            Siren(W, W, w0=1.0, is_first=False),
            Siren(W, W, w0=1.0, is_first=False),
            Siren(W, W, w0=1.0, is_first=False),
        )
        self.layer1 = nn.Sequential(
            Siren(W, W, w0=1.0, is_first=False),
            Siren(W, W, w0=1.0, is_first=False),
            Siren(W, W, w0=1.0, is_first=False),
            Siren(W, W, w0=1.0, is_first=False),
        )
        self.xyz_encoding_final = DenseLayer(W, W, 'linear')

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            Siren(W+in_channels_dir, W//2, w0=1.0, is_first=False),
        )

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            DenseLayer(W//2, 3, 'linear'), nn.Sigmoid(),
        )

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        x = self.layer0(input_xyz)
        x = self.layer1(x)
        sigma = self.sigma(x)
        
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(x)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], dim=-1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        
        out = torch.cat([rgb, sigma], -1)

        return out