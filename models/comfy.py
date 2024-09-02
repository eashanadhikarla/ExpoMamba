"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn

def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False

def is_device_mps(device):
    return is_device_type(device, 'mps')

def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False #pytorch bug? mps doesn't support non blocking
    return True

def cast_bias_weight(s, input):
    bias = None
    non_blocking = device_supports_non_blocking(input.device)
    if s.bias is not None:
        bias = s.bias.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
        if s.bias_function is not None:
            bias = s.bias_function(bias)
    weight = s.weight.to(device=input.device, dtype=input.dtype, non_blocking=non_blocking)
    if s.weight_function is not None:
        weight = s.weight_function(weight)
    return weight, bias

class CastWeightBiasOp:
    comfy_cast_weights = False
    weight_function = None
    bias_function = None

class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)


class Block(nn.Module):
    def __init__(self, n_in, n_out, ks=3, pd=1, b=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pd, bias=b),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size=ks, padding=pd, bias=b),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, kernel_size=ks, padding=pd, bias=b),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv_block(x) + self.skip(x))


# def conv(n_in, n_out, **kwargs):
#     return disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

# class Block(nn.Module):
#     def __init__(self, n_in, n_out, ks=3, pd=1, b=False):
#         super().__init__()
#         self.conv = nn.Sequential(
#             # disable_weight_init.Conv2d(n_in, n_out, 3, padding=1), # conv(n_in, n_out),
#             nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pd, groups=n_in, bias=b),
#             nn.ReLU(),
#             # disable_weight_init.Conv2d(n_in, n_out, 3, padding=1), # conv(n_out, n_out),
#             nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pd, groups=n_in, bias=b),
#             nn.ReLU(),
#             # disable_weight_init.Conv2d(n_in, n_out, 3, padding=1), #conv(n_out, n_out)
#             nn.Conv2d(n_in, n_out, kernel_size=ks, padding=pd, groups=n_in, bias=b),
#         )
#         self.skip = disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
#         self.fuse = nn.ReLU()
#     def forward(self, x):
#         return self.fuse(self.conv(x) + self.skip(x))
