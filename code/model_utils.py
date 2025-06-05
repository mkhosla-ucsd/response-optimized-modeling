from torch import nn
import numpy as np
import math
from torch.nn import functional as F

def positive(weight):
    """
    Enforces tensor to be positive. Changes the tensor in place. Produces no gradient.
    Args:
        weight: tensor to be positive
    """
    
    weight.data *= weight.data.ge(0).float()
    
    
class Encoder(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       

    def forward(self, x, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        x = self.core(x)
        if detach_core:
            x = x.detach()
        if "sample" in kwargs:
            x = self.readout(x,  sample=kwargs["sample"])
        else:
            x = self.readout(x)
        return x 
    

class Encoder_semantic(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
       

    def forward(self, x, data_key=None, detach_core=False, fake_relu = False, **kwargs):
        feats = self.core(x)
        if detach_core:
            feats = feats.detach()
        
        x = self.readout(feats, x)
        return x 
    
    
