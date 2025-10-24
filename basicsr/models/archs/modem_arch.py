# Code Implementation of the MambaIR Model
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat


def batch_bitwise_interleave(x, y):
    """Optimized bitwise interleave for 32-bit integers"""
    x = (x | (x << 16)) & 0x0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555

    y = (y | (y << 16)) & 0x0000FFFF
    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555

    z = x | (y << 1)
    return z


def morton_indices(H, W, reverse=False):
    """Generate Z-order curve indices for a 2D grid of shape (H, W) efficiently."""
    # Create grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W, dtype=np.uint32), np.arange(H, dtype=np.uint32))

    if reverse:
        # Swap x and y coordinates for reversed Z-order
        x_coords, y_coords = y_coords, x_coords

    # Interleave the bits of the coordinates
    z_indices = batch_bitwise_interleave(x_coords, y_coords)

    # Flatten the grid and sort by Z-order index
    z_order = np.argsort(z_indices.flatten())

    return z_order


def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')


def to_3d(x):
    #    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c').contiguous()


def to_4d(x, h, w):
    #    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:  # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat // compress_ratio, num_feat // compress_ratio, 3, 1, 1,
                          groups=num_feat // compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3, 1, padding=2, groups=num_feat, dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class ModemSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.deg_params = kwargs.get("deg_params", None)
        self.direction = kwargs.get("direction", None)
        if self.direction is None:
            raise ValueError("direction must be specified")

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        if self.deg_params is None:
            self.x_proj = (
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            )
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
            del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        if self.deg_params is not None:
            self.deg_scale = nn.Sequential(
                nn.Linear(self.deg_params, self.d_inner, bias=False, **factory_kwargs)
            )
            self.deg_bias = nn.Sequential(
                nn.Linear(self.deg_params, self.d_inner, bias=False, **factory_kwargs)
            )
            self.kernel1 = nn.Sequential(
                nn.Linear(self.deg_params, self.d_inner, bias=False, **factory_kwargs)
            )
            self.kernel2 = nn.Sequential(
                nn.Linear(self.deg_params, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            )
            self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, deg=None):
        B, C, H, W = x.shape  # 1, 192, 64, 64
        L = H * W
        K = 1

        if self.deg_params is not None:
            fv, kernel = deg[0], deg[1]
            deg_scale = self.deg_scale(fv).view(B, C, 1, 1).expand_as(x)
            deg_bias = self.deg_bias(fv).view(B, C, 1, 1).expand_as(x)
            x = x * deg_scale + deg_bias

        if self.direction == 'wh':
            x_wh_indices = morton_indices(H, W, reverse=False)
            x_wh = x.view(B, C, H * W)[:, :, x_wh_indices]
            xs = x_wh.view(B, K, -1, L).contiguous()  # (b, k, d, l)
        else:
            x_hw_indices = morton_indices(H, W, reverse=True)
            x_hw = x.view(B, C, H * W)[:, :, x_hw_indices]
            xs = x_hw.view(B, K, -1, L).contiguous()

        if self.deg_params is not None:
            kernel = self.kernel1(kernel)
            kernel = self.kernel2(kernel.permute(0, 2, 1))
            kernel = self.softmax(kernel.permute(0, 2, 1) / math.sqrt(self.d_inner))
            kernel = kernel.unsqueeze(1)

            x_dbl = torch.einsum("b k d l, b k c d -> b k c l", xs.view(B, K, -1, L), kernel)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L),
                                 self.x_proj_weight)  # 1, 4, 38, 4096

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)  # 1, 4, 192, 4096
        xs = xs.float().view(B, -1, L)  # 1, 768, 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) 1, 768, 4096

        Bs = Bs.float().view(B, K, -1, L)  # 1, 4, 16, 4096
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) 1, 4, 16, 4096
        Ds = self.Ds.float().view(-1)  # 768
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # 768, 16
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)  768
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.direction == 'wh':
            y_wh_indices_inv = np.argsort(x_wh_indices)
            y_wh = out_y[:, 0][:, :, y_wh_indices_inv].contiguous()
            y = y_wh
        else:
            y_hw_indices_inv = np.argsort(x_hw_indices)
            y_hw = out_y[:, 0][:, :, y_hw_indices_inv].contiguous()
            y = y_hw

        y = y.view(B, C, H, W).contiguous()

        return y

    def forward(self, x: torch.Tensor, x_size, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.act(self.conv2d(to_4d(x, x_size[0], x_size[1])))
        deg = kwargs.get("deg", None)
        y = self.forward_core(x, deg)
        y = self.out_norm(to_3d(y))
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ModemBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.modem_ss2d = ModemSS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                    **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size, **kwargs):
        # x [B,HW,C]
        B, L, C = input.shape
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.modem_ss2d(x, x_size, **kwargs))
        x = x * self.skip_scale2 + to_3d(self.conv_blk(to_4d(self.ln_2(x), x_size[0], x_size[1])))
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 **kwargs):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:
                direction = 'wh'
            else:
                direction = 'hw'

            self.blocks.append(ModemBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                is_light_sr=is_light_sr,
                direction=direction,
                **kwargs))

    def forward(self, x, x_size, **kwargs):
        for blk in self.blocks:
            x = blk(x, x_size, **kwargs)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr=False,
                 **kwargs):
        super(ResidualGroup, self).__init__()

        self.dim = dim

        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
            **kwargs)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, **kwargs):
        return self.conv(to_4d(self.residual_group(to_3d(x), x_size, **kwargs), x_size[0], x_size[1])) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9

        return flops


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ModemIR(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super(ModemIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.down_layer1 = ResidualGroup(
            dim=embed_dim,
            depth=depths[0],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )
        self.downsample1_2 = Downsample(embed_dim)

        self.down_layer2 = ResidualGroup(
            dim=embed_dim * 2,
            depth=depths[1],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )
        self.downsample2_3 = Downsample(embed_dim * 2)

        self.down_layer3 = ResidualGroup(
            dim=embed_dim * 2 * 2,
            depth=depths[2],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )
        self.downsample3_4 = Downsample(embed_dim * 2 * 2)

        self.latent = ResidualGroup(
            dim=embed_dim * 2 * 4,
            depth=depths[3],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )

        self.upsample4_3 = Upsample(embed_dim * 2 * 4)
        self.reduce3 = nn.Conv2d(embed_dim * 2 * 4, embed_dim * 2 * 2, 1, 1, 0)
        self.up_layer3 = ResidualGroup(
            dim=embed_dim * 2 * 2,
            depth=depths[4],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:4]):sum(depths[:4 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )

        self.upsample3_2 = Upsample(embed_dim * 2 * 2)
        self.reduce2 = nn.Conv2d(embed_dim * 2 * 2, embed_dim * 2, 1, 1, 0)
        self.up_layer2 = ResidualGroup(
            dim=embed_dim * 2 * 1,
            depth=depths[5],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:5]):sum(depths[:5 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )

        self.upsample2_1 = Upsample(embed_dim * 2 * 1)
        self.up_layer1 = ResidualGroup(
            dim=embed_dim * 2 * 1,
            depth=depths[6],
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:6]):sum(depths[:6 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )

        # self.norm = norm_layer(self.num_features)
        self.refinement = ResidualGroup(
            dim=embed_dim * 2 * 1,
            depth=4,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            drop_path=0.,  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            is_light_sr=self.is_light_sr,
            **kwargs
        )

        self.conv_last = nn.Conv2d(embed_dim * 2 * 1, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, **kwargs):
        x_size = (x.shape[2], x.shape[3])

        x = self.pos_drop(x)

        x_down1 = self.down_layer1(x, x_size, **kwargs)
        x_down2 = self.downsample1_2(x_down1)  # B, 2C, H/2, W/2
        x_size = (x_down2.shape[2], x_down2.shape[3])

        x_down2 = self.down_layer2(x_down2, x_size, **kwargs)
        x_down3 = self.downsample2_3(x_down2)  # B, 4C, H/4, W/4
        x_size = (x_down3.shape[2], x_down3.shape[3])

        x_down3 = self.down_layer3(x_down3, x_size, **kwargs)
        x_down4 = self.downsample3_4(x_down3)  # B, 8C, H/8, W/8
        x_size = (x_down4.shape[2], x_down4.shape[3])

        x_latent = self.latent(x_down4, x_size, **kwargs)

        x_up3 = self.upsample4_3(x_latent)  # B, 4C, H/4, W/4
        x_size = (x_up3.shape[2], x_up3.shape[3])
        x_up3 = torch.cat([x_up3, x_down3], dim=1)  # B, 8C, H/4, W/4
        x_up3 = self.reduce3(x_up3)
        x_up3 = self.up_layer3(x_up3, x_size, **kwargs)

        x_up2 = self.upsample3_2(x_up3)  # B, 4C, H/2, W/2
        x_size = (x_up2.shape[2], x_up2.shape[3])
        x_up2 = torch.cat([x_up2, x_down2], dim=1)  # B, 6C, H/2, W/2
        x_up2 = self.reduce2(x_up2)
        x_up2 = self.up_layer2(x_up2, x_size, **kwargs)

        x_up1 = self.upsample2_1(x_up2)  # B, 3C, H, W
        x_size = (x_up1.shape[2], x_up1.shape[3])
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # B, 5C, H, W
        x_up1 = self.up_layer1(x_up1, x_size, **kwargs)
        x_size = (x_up1.shape[2], x_up1.shape[3])
        x = self.refinement(x_up1, x_size, **kwargs)

        return x.contiguous()

    def forward(self, x, **kwargs):
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range
        inp = x
        x = self.conv_first(x)
        x = self.forward_features(x, **kwargs)
        x = self.conv_last(x)
        x = x + inp
        # x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ModemIDE(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 n_feats=180,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(ModemIDE, self).__init__()
        num_in_ch = in_chans
        self.img_range = img_range
        if in_chans == 333:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 6-layer
            if i_layer % 2 == 0:
                direction = 'wh'
            else:
                direction = 'hw'

            layer = ModemBlock(
                hidden_dim=embed_dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                is_light_sr=self.is_light_sr,
                direction=direction,
                **kwargs
            )
            self.layers.append(layer)
        # self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, n_feats * 4),
            nn.SiLU(),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.SiLU()
        )
        self.compress = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats),
            nn.SiLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.kernel1 = nn.Sequential(
            nn.Conv2d(embed_dim, n_feats, 3, 1, 1, bias=False),
        )
        self.kernel2 = nn.Sequential(
            nn.Conv2d(embed_dim, n_feats, 3, 1, 1, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = to_3d(self.pos_drop(x))

        for layer in self.layers:
            x = layer(x, x_size)

        return to_4d(x, x_size[0], x_size[1])

    def forward(self, x):
        B, C, H, W = x.shape
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        # x = x / self.img_range + self.mean
        fm = x
        fv = self.avgpool(x).view(x.size(0), -1)

        fea1 = self.mlp(fv)
        kl_fea = [fea1]
        fv = self.compress(fea1)

        kernel1 = self.kernel1(fm).view(B, -1, H * W)
        kernel2 = self.kernel2(fm).view(B, -1, H * W)
        kernel = kernel2 @ kernel1.transpose(1, 2)

        return fv, kernel, kl_fea


class ModemDeg(nn.Module):
    def __init__(self, n_feats=36, n_encoder_res=[1, 1], scale=4, n_sr_blocks=[4, 4, 6, 8, 6, 4, 4]):
        super(ModemDeg, self).__init__()

        # Generator
        self.G = ModemIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.,
            d_state=16,
            depths=n_sr_blocks,
            embed_dim=n_feats,
            mlp_ratio=2.,
            deg_params=96,
        )

        self.E = ModemIDE(
            in_chans=6,
            img_size=64,
            img_range=1.,
            d_state=16,
            depths=n_encoder_res,
            embed_dim=96,
            n_feats=96,
            mlp_ratio=2.,
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x, gt=None):
        if gt is not None:
            fv, fm, kl_fea = self.E(torch.cat([x, gt], dim=1))
            deg_repre = [fv, fm]
            sr = self.G(x, deg=deg_repre)

        else:
            fv, fm, kl_fea = self.E(x)
            deg_repre = [fv, fm]
            sr = self.G(x, deg=deg_repre)

        if self.training:
            return sr, kl_fea
        else:
            return sr


class Modem(nn.Module):
    def __init__(self, n_feats=36, n_encoder_res=[1, 1], scale=4, n_sr_blocks=[4, 4, 6, 8, 6, 4, 4]):
        super(Modem, self).__init__()

        # Generator
        self.G = ModemIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            img_range=1.,
            d_state=16,
            depths=n_sr_blocks,
            embed_dim=n_feats,
            mlp_ratio=2.,
            deg_params=96,
        )

        self.E = ModemIDE(
            in_chans=3,
            img_size=64,
            img_range=1.,
            d_state=16,
            depths=n_encoder_res,
            embed_dim=96,
            n_feats=96,
            mlp_ratio=2.,
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x):
        fv, fm, kl_fea = self.E(x)
        deg_repre = [fv, fm]
        sr = self.G(x, deg=deg_repre)

        if self.training:
            return sr, kl_fea
        else:
            return sr


if __name__ == '__main__':
    device = 'cuda:0'
    model = Modem().to(device)
    print(model)
    lr = torch.randn(1, 3, 128, 128).to(device)
    sr, _ = model(lr)
    print(sr.shape)
    count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters)
