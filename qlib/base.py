"""
Base quanitzation module
"""

from typing import Union
import torch as th
from torch import Tensor
from torch.nn.common_types import _size_2_t
import torch.nn.functional as thf
from qlib.utils import AverageMeter
from .utils import plot_masks

class CenterCrop2d(th.nn.Module):
    def __init__(self, crop_px: int = 0):
        super().__init__()
        self.crop = crop_px
    def forward(self, x):                       # symmetrical crop
        if self.crop == 0:
            return x
        return x[:, :, self.crop:-self.crop, self.crop:-self.crop]

class QBase(th.nn.Module):
    """
    Base parent class for quantization method design
    
    Args:
    - nbit: int
    Precision of the quantization target (e.g., weight, activation, bias)

    deploy: bool
    Post quantization deployment
    """
    def __init__(self, nbit:int=8) -> None:
        super().__init__()
        self.nbit = nbit
        self.deploy = False

        self.register_buffer("scale", th.tensor(1.0))
        self.register_buffer("offset", th.tensor(0.0))

    def q(self, x:th.Tensor):
        return x
    
    def train_func(self, x:th.Tensor):
        return self.q(x)
    
    def eval_func(self, x:th.Tensor):
        return self.train_func(x)
    
    def forward(self, x:th.Tensor):
        if not self.deploy:
            y = self.train_func(x)
        else:
            y = self.train_func(x)
        return y
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f"nbit={self.nbit}"
    
class QConv2d(th.nn.Conv2d):
    def __init__(self, in_channels: int, 
            out_channels: int, 
            kernel_size: _size_2_t, 
            stride: _size_2_t = 1, 
            padding: _size_2_t = 0, 
            dilation: _size_2_t = 1, 
            groups: int = 1, 
            bias: bool = True, 
            padding_mode: str = 'zeros', 
            device=None, 
            dtype=None
        ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.wbit = 8
        self.abit = 8

        # quantizer
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()
        self.bq = th.nn.Identity()
    
    def forward(self, x: Tensor, ptq_training=False, mask_path=None, n=None, idx=None) -> Tensor:
        wq = self.wq(self.weight)
        if isinstance(self.bq, Union[th.nn.Identity]):
            xq = self.xq(x)
        else:
            if ptq_training and mask_path is not None:
                        comming_max = th.amax(x-self.bq.smooth_shift.unsqueeze(0), dim=(1), keepdim=True).squeeze(1)
                        comming_min = th.amin(x-self.bq.smooth_shift.unsqueeze(0), dim=(1), keepdim=True).squeeze(1)  
                        plot_masks(comming_max, path=mask_path + n +str(idx) +"_smax.png")
                        plot_masks(comming_min, path=mask_path + n +str(idx) +"_smin.png")
            xq = self.xq(x-self.bq.smooth_shift.unsqueeze(0))
        
        # convolution
        output = thf.conv2d(
            xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        
        if not isinstance(self.bq, Union[th.nn.Identity]):

            output = self.bq(output, train_flag=ptq_training)
        return output
    

class QLinear(th.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.wbit = 8
        self.abit = 8

        # quantizer
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        
        wq = self.wq(self.weight)
        xq = self.xq(input)

        output = thf.linear(xq, wq, self.bias)
        return output
    

class QConv2dWN(th.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(QConv2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )
        self.g = th.nn.Parameter(th.ones(out_channels))

        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()

    def forward(self, x:th.Tensor):
        wnorm = th.sqrt(th.sum(self.weight**2))
        wn = self.weight * self.g[:, None, None, None] / wnorm,
        
        # quantize
        xq = self.xq(x)
        wq = self.wq(wn)

        return thf.conv2d(
            xq,
            wq,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
    
class QLinearWN(th.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QLinearWN, self).__init__(in_features, out_features, bias)
        self.g = th.nn.Parameter(th.ones(out_features))

        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()
    
    def forward(self, input):
        wnorm = th.sqrt(th.sum(self.weight**2))
        wn = self.weight * self.g[:, None] / wnorm

        # quantize
        xq = self.xq(input)
        wq = self.wq(wn)

        return thf.linear(xq, wq, self.bias)


class QPixelConv2dWN(th.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        
        self.g = th.nn.Parameter(th.ones(out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride       = stride[0]
        self.kernel_size  = kernel_size[0]
        self.k_low        = kernel_size[0]  //  self.stride       # conv_lr.kernel_size
        self.padding_low  = padding[0]
        # TODO: add support for geoups>1
        assert dilation[0] == 1, "dilation must be 1"
        assert groups == 1, "groups must be 1"
        # print(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print(self.in_channels, self.out_channels * self.stride**2)
        self.conv_lr = th.nn.Conv2d(
            self.in_channels, self.out_channels * self.stride**2,
            kernel_size=self.k_low, stride=1, padding=self.k_low - self.padding_low,
            bias=bias)

        self.pix  = th.nn.PixelShuffle(self.stride)
        self.crop = CenterCrop2d(self.k_low - self.padding_low)          # while p_low>0, crop the edge
        # self.seq  = th.nn.Sequential(self.conv_lr, self.pix, self.crop)

    @th.no_grad()
    def build_from_params(self,
                          weight_t: th.Tensor,   # (C_in, C_out, kT, kT)
                          g_vec:     th.Tensor,  # (C_out,)
                          bias_t:    th.Tensor = None):
        s      = self.stride
        C_in   = self.in_channels
        C_out  = self.out_channels
        k_low  = self.k_low
        kT     = weight_t.shape[-1]

        assert weight_t.shape == (C_in, C_out, kT, kT)
        assert kT == k_low * s                 
        assert g_vec.shape == (C_out,)              


        self.g.copy_(g_vec)
        w_norm = th.sqrt(th.sum(weight_t**2))
        W_hat  = weight_t * g_vec[None,:,None,None] / w_norm          # (C_in,C_out,kT,kT)

        # write conv_lr.weight —— 180° flip + sparsity
        for cout in range(C_out):
            for cin in range(C_in):
                for i0 in range(s):              # i0,j0 ∈ 0..s-1
                    for j0 in range(s):
                        for u in range(k_low):   # u,v ∈ 0..k_low-1
                            for v in range(k_low):
                                kH = (k_low - 1 - u) * s + i0   # flip & sparse idx
                                kW = (k_low - 1 - v) * s + j0
                                q  = cout * s**2 + i0 * s + j0
                                self.conv_lr.weight[q, cin, u, v] = W_hat[cin, cout, kH, kW]

        if bias_t is not None and self.conv_lr.bias is not None:
            assert bias_t.shape == (C_out,), "bias shape mismatch"
            self.conv_lr.bias.view(C_out, s**2).copy_(
                bias_t.view(-1,1).expand(-1, s**2))

    def forward(self, x):
        conv = self.conv_lr(x)

        ps = self.pix(conv)

        out = self.crop(ps)
        
        return out
    
class QConvTranspose2dWN(th.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(QConvTranspose2dWN, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smooth = False
        print(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.g = th.nn.Parameter(th.ones(out_channels))
        self.in_channels = in_channels
        # quantizers
        self.wq = th.nn.Identity()
        self.xq = th.nn.Identity()
        self.bq = th.nn.Identity()
        self.frame_diff = AverageMeter()
        self.prev_frame = None

    def set_smooth(
    self,
    alpha: float = 0.5,
    cached_data=None,
    weight=None,
    k_percent: float = 5.0,  
):

        device = self.weight.device
        C       = self.in_channels
        eps     = 1e-8

        x_max   = th.zeros(C, device=device)
        x_var   = th.zeros(C, device=device)

        x_smooth = th.zeros(self.in_channels).to(self.weight.device)
        if cached_data is not None:
            n_samples = 0
            for x, _ in cached_data:
                x = x.to(device).abs()
                if weight is not None:
                    x = x * weight.to(device)

                cur_max = x.amax(dim=(0, 2, 3))
                x_max   = th.maximum(x_max, cur_max)


                n_samples += x.shape[0]
                x_var += x.var(dim=(0, 2, 3), unbiased=False) * x.shape[0]

            x_var /= max(n_samples, 1)

            w_max = self.weight.abs().amax(dim=(1, 2, 3))
            x_smooth = (x_max.clamp_min(eps).pow(alpha) /
                    (w_max.clamp_min(eps).pow(1 - alpha)))

            k = max(0, int(round(C * k_percent / 100.0)))
            topk_idx = th.topk(x_var, k, largest=True).indices
            x_smooth[topk_idx] = 1.0      

            x_smooth = x_smooth.clamp_min(eps)

            factor = x_smooth.view(-1, 1, 1, 1)
            with th.no_grad():
                self.weight.mul_(factor)
            
            print(x_smooth.max(),x_smooth.min())
            print(x_smooth)
            print(x_var)

        self.smooth    = True
        self.x_smooth  = th.nn.Parameter(x_smooth.detach())



    def forward(self, x, ptq_training=False, mask_path=None, n=None, idx=None, pixel=False):
        wnorm = th.sqrt(th.sum(self.weight**2))

        wn = self.weight * self.g[None, :, None, None] / wnorm
        if self.smooth:
            x = x / self.x_smooth.view(1, -1, 1, 1)
        # quantize
        if isinstance(self.bq, Union[th.nn.Identity]):
            xq = self.xq(x)
        else:
            xq = self.xq(x-(self.bq.smooth_shift*self.bq.delta).unsqueeze(0))

        wq = self.wq(wn)


        yq = thf.conv_transpose2d(
            xq,
            wq,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if not isinstance(self.bq, Union[th.nn.Identity]):

            yq = self.bq(yq, train_flag=ptq_training)
        
        return yq

