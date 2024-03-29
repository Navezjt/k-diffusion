import torch
from torch import FloatTensor
from torch.nn import Module
from typing import Sequence

class Normalize(Module):
    mean: FloatTensor
    std: FloatTensor
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(-1, 1, 1))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return (x - self.mean) / self.std

    def forward_(self, x: FloatTensor) -> FloatTensor:
        x.sub_(self.mean)
        x.div_(self.std)
        return x

    def inverse(self, x: FloatTensor) -> FloatTensor:
        return x * self.std + self.mean
    
    def inverse_(self, x: FloatTensor) -> FloatTensor:
        x.mul_(self.std)
        x.add_(self.mean)
        return x

# convenient for consumers such as Sequential, which only use forward()
class Normalize_(Module):
    mean: FloatTensor
    std: FloatTensor
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(-1, 1, 1))

    def forward(self, x: FloatTensor) -> FloatTensor:
        x.sub_(self.mean)
        x.div_(self.std)
        return x