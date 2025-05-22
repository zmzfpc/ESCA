

import torch
from qlib.quant import ActQuantizer


class ActQuant(torch.nn.Module):
    def __init__(self, nbit: int=8, gp = -1, sym: bool=False, cr: float=1.0):
        super().__init__()
        self.quantizer = ActQuantizer()
        self.quantizer.configure(bits=nbit, groupsize=gp, sym=sym, clip_ratio=cr)
        
    def forward(self, x):
        x_dtype = x.dtype

        if self.quantizer.bits < 32:
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()   
        
        return x


    


