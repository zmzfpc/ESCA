

import math
import time

import torch
import torch.nn as nn

from typing import Union, List
from qlib.quant import *
from qlib.base import QConvTranspose2dWN, QConv2d
from models import Conv2dWN

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def downsample_mask(uv:torch.Tensor, height):
        # stride and window
    kernel = uv.size(2) // height
    mask = torch.max_pool2d(uv, kernel_size=kernel, stride=kernel)
    return mask.cuda()

def shape_wise_filter(y:torch.Tensor, tau:float=0.5):

    assert len(y.size()) == 4, "The shape of the feature map tensor must be 4-D"

    num_pixel_keep = int(y.size(2)*y.size(3) * tau)

    b, c, h, w = y.size()
    yd = y.detach()

    with torch.no_grad():
        masks = []

            # row channel plane
        ystd = yd.std(dim=[1])

        for i in range(b):
            mask = torch.zeros(h, w)
            scores = ystd[i]

                # top-k score
            tpk_score = torch.topk(scores.view(-1), num_pixel_keep, sorted=True)
            threshold = tpk_score.values[-1]

                # mask[scores.lt(threshold), :] = 1.0
            mask = scores.lt(threshold).float()
            masks.append(mask.unsqueeze(0).unsqueeze(0))
        
    return torch.cat(masks, dim=0).cuda()
    
class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, Union[nn.Conv2d, QConvTranspose2dWN, Conv2dWN]):
            W = W.permute([1, 0, 2, 3])
            W = W.flatten(1)

        if isinstance(self.layer, nn.Conv1d):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, uv, shift, tau=0.1):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if shift is not None:
            inp = inp - shift.unsqueeze(0)

        if uv is not None:
            if tau > 0:
                mask = downsample_mask(uv, height=inp.size(2)).mul(shape_wise_filter(inp, tau=tau))
            else:
                mask = downsample_mask(uv, height=inp.size(2))
        # filter_mask = shape_wise_filter(inp, tau=tau)
        # mask = mask.mul(filter_mask)
            inp = inp.mul(mask)
            if DEBUG:
                print('mask', mask.shape)
                print('inp', inp.shape)

        

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, nn.Conv1d):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, Union[nn.Conv2d, QConvTranspose2dWN, Conv2dWN]):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):  
        
        W = self.layer.weight.data.clone()


        if isinstance(self.layer, Union[nn.Conv2d, QConvTranspose2dWN, Conv2dWN]):
            W = W.permute([1, 0, 2, 3])
            W = W.flatten(1)
        if isinstance(self.layer, nn.Conv1d):
            W = W.t()
        W = W.float()
        if DEBUG:
            W_debug = W.clone()
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        self.H = self.H.cpu()
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)


        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


            if DEBUG:
                W_debug[:, :i2] = Q[:, :i2]
                W_debug[:, i2:] = W[:, i2:]
                self.layer.weight.data = W_debug.reshape(self.layer.weight.data.permute([1, 0, 2, 3]).shape).to(self.layer.weight.data.dtype).permute([1, 0, 2, 3])
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print("MSEloss", nn.MSELoss()(self.layer(self.inp1), self.out1))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv1d):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.data.permute([1, 0, 2, 3]).shape).to(self.layer.weight.data.dtype).permute([1, 0, 2, 3])
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            print("MSEloss", nn.MSELoss()(self.layer(self.inp1), self.out1))
        
        if torch.any(torch.isnan(self.layer.weight.data)):
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')
        return self.layer

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
