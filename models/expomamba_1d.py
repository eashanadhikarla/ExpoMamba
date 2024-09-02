import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from mamba_ssm import Mamba
from vmamba import VSSM


class ComplexConv2d(nn.Module):
    '''https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, complex_input):
        real = complex_input.real
        imag = complex_input.imag
        return torch.complex(self.real_conv(real) - self.imag_conv(imag),
                             self.real_conv(imag) + self.imag_conv(real))

class DynamicAmplitudeScaling(nn.Module):
    def __init__(self, channel):
        super(DynamicAmplitudeScaling, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, channel, 1, 1))

    def forward(self, amp):
        scaled_amp = amp * self.scale
        return scaled_amp

class PhaseContinuityLayer(nn.Module):
    def __init__(self, channel):
        super(PhaseContinuityLayer, self).__init__()
        # Define a convolutional layer with a large kernel to enforce smoothness over a larger spatial region
        # The kernel size and other parameters would be chosen based on the specific requirements of the task
        self.smoothing_conv = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, groups=channel, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to scale the phase adjustments softly

        # Initialize the convolutional weights to act as a smoothing kernel (like a Gaussian blur)
        nn.init.constant_(self.smoothing_conv.weight, 1/25)  # For a 5x5 kernel, sum should be 1

    def forward(self, phase):
        # Apply the smoothing convolution
        smoothed_phase = self.smoothing_conv(phase)
        # Apply sigmoid to scale the phase adjustments
        scaled_phase = self.sigmoid(smoothed_phase)
        return scaled_phase


class FSSB(nn.Module):
    def __init__(self, c=3, reduced_dim=None): # w=128, h=128,
    # def __init__(self, c=3, w=128, h=128, reduced_dim=None):
        super(FSSB, self).__init__()

        self.reduced_dim = reduced_dim
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(), # ReLU(),
            nn.Conv2d(in_channels=4, out_channels=c, kernel_size=3, stride=1, padding=1)
        )
        self.complex_conv = ComplexConv2d(c, c, kernel_size=3, stride=1, padding=1)

        # Simplified Mamba models
        self.model_amp = Mamba(d_model=c, d_state=16, d_conv=4, expand=1)
        self.model_pha = Mamba(d_model=c, d_state=16, d_conv=4, expand=1)

        # Optimized 1D convolutional layers
        self.con1_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
        self.con1_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

        # Cross modulation layers adapted to reduced dimensions
        self.cross_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
        self.cross_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

        # Remaining layers
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=(c))
        # self.bn = nn.BatchNorm2d(c)
        self.softmax = nn.Softmax(dim=1)

        self.amplitude_scaling = DynamicAmplitudeScaling(c)
        self.phase_continuity = PhaseContinuityLayer(c)

    def forward(self, x):
        # Compute FFT to get amplitude and phase
        fft_x = torch.fft.fft2(x)
        amp = torch.real(fft_x)
        pha = torch.imag(fft_x)

        # Apply Dynamic Amplitude Scaling and Phase Continuity
        amp_scaled = self.amplitude_scaling(amp)
        pha_continuous = self.phase_continuity(pha)

        # Processing with complex convolution
        complex_input = torch.complex(amp_scaled, pha_continuous)
        complex_processed = self.complex_conv(complex_input)

        # Separate processed amplitude and phase
        processed_amp = torch.real(complex_processed)
        processed_pha = torch.imag(complex_processed)

        # Flatten for layer normalization and 1D convolutions
        b, c, w, h = processed_amp.shape
        flat_dim = w * h
        processed_amp_flat = processed_amp.view(b, c, flat_dim).permute(0, 2, 1)
        processed_pha_flat = processed_pha.view(b, c, flat_dim).permute(0, 2, 1)

        amp_ln = self.ln(processed_amp_flat)
        pha_ln = self.ln(processed_pha_flat)

        # Process amplitude and phase with Mamba models
        amp_mamba = self.model_amp(amp_ln)
        pha_mamba = self.model_pha(pha_ln)

        # Apply 1D convolutions
        amp_conv = self.con1_1(amp_mamba.reshape(b, self.reduced_dim, -1))
        pha_conv = self.con1_2(pha_mamba.reshape(b, self.reduced_dim, -1))

        # Apply softmax for normalization
        amp_softmax = self.softmax(amp_conv)
        pha_softmax = self.softmax(pha_conv)

        # Apply cross modulation
        attention_amp = self.cross_1(amp_softmax) * self.cross_2(pha_softmax)
        attention_pha = self.cross_1(pha_softmax) * self.cross_2(amp_softmax)

        # Reshape back to original dimensions and apply attention
        amp_out = amp_conv.view(b, c, w, h) + attention_amp.view(b, c, w, h)
        pha_out = pha_conv.view(b, c, w, h) + attention_pha.view(b, c, w, h)

        # Combine amplitude and phase, and apply inverse FFT
        # output_fft = torch.complex(amp_out * torch.cos(pha_out), amp_out * torch.sin(pha_out))
        output_fft = torch.polar(amp_out, pha_out)
        output = torch.fft.ifft2(output_fft).real

        # Apply final smoothing convolution
        output = self.smooth(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConv(nn.Module):
    """(depthwise-convolution => [BN] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.fssb = FSSB(c=in_channels, reduced_dim=32) # w=128, h=128,
        # self.ub = FSSB(c=in_channels, reduced_dim=256) # w=128, h=128,

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        inputs = F.interpolate(x, size=[32, 32], mode='bilinear', align_corners=True)
        # print("Before fssb block: ", inputs.shape)
        outputs = self.fssb(inputs)
        outputs = F.interpolate(outputs, size=[x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True) + x
        return self.double_conv(outputs)


class Down(nn.Module):
    """Downscaling with maxpool then double conv and concatenate an additional input tensor"""
    
    def __init__(self, in_channels, out_channels, merge_mode='concat'):
        super(Down, self).__init__()
        self.merge_mode = merge_mode
        
        # The DoubleConv module might need adjustment based on the merge_mode
        if merge_mode == 'concat':
            # When concatenating, the number of input channels to the second convolution doubles
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels, in_channels * 2)
            )
        else:
            # No concatenation; keep as is
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x1, x2=None):
        x1 = self.maxpool_conv(x1)

        if self.merge_mode == 'concat' and x2 is not None:
            # Ensuring dimensionality match for concatenation
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            
            # Concatenate along the channel dimension
            x = torch.cat([x2, x1], dim=1)
            return x
        else:
            return x1


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ExpoMamba(nn.Module):
    """ 
    Full assembly of the parts to form the complete network 
    -------
    The proposed network consists of two CNNs: the first part is to use the first-order 
    unfolding Taylor’s formula to build an interpretable network, and combine two UNets 
    in the form of first-order Taylor’s polynomials. Then we use this constructed network 
    to extract the feature maps of the low-resolution input image, and finally process 
    the feature maps to form a multi-dimensional tensor termed a bilateral grid that acts 
    on the original image to yield an enhanced result. The second part is the image 
    enhancement using the bilateral grid. In addition, we propose a polynomial channel 
    enhancement method to enhance UHD images.
    """
    def __init__(self, n_channels, bilinear=True, base_channels=32):
        super(ExpoMamba, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        # Initial convolution block
        self.inc = DoubleConv(n_channels, base_channels)
        # Downsampling path
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, (base_channels * 16//factor))

        # Upsampling path
        self.up1 = Up( (base_channels * 16), (base_channels * 8) // factor, bilinear)
        self.up2 = Up((base_channels * 8), (base_channels * 4) // factor, bilinear)
        self.up3 = Up((base_channels * 4), (base_channels * 2) // factor, bilinear)
        self.up4 = Up((base_channels * 2), base_channels, bilinear)
        self.outc = OutConv(base_channels, n_channels)

    def forward(self, inp):
        x = inp
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) + inp
        return x




# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import torchinfo
# from mamba_ssm import Mamba

# class ComplexConv2d(nn.Module):
#     '''https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
#     '''
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
#         super(ComplexConv2d, self).__init__()
#         self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

#     def forward(self, complex_input):
#         real = complex_input.real
#         imag = complex_input.imag
#         return torch.complex(self.real_conv(real) - self.imag_conv(imag),
#                              self.real_conv(imag) + self.imag_conv(real))

# class DynamicAmplitudeScaling(nn.Module):
#     def __init__(self, channel):
#         super(DynamicAmplitudeScaling, self).__init__()
#         self.scale = nn.Parameter(torch.ones(1, channel, 1, 1))

#     def forward(self, amp):
#         scaled_amp = amp * self.scale
#         return scaled_amp

# # class PhaseContinuityLayer(nn.Module):
# #     def __init__(self, channel):
# #         super(PhaseContinuityLayer, self).__init__()
# #         # Placeholder for operations that enhance phase continuity.
# #         # This example uses an identity operation for simplicity.
# #         self.phase_op = nn.Identity()

# #     def forward(self, pha):
# #         continuous_pha = self.phase_op(pha)
# #         return continuous_pha

# class PhaseContinuityLayer(nn.Module):
#     def __init__(self, channel):
#         super(PhaseContinuityLayer, self).__init__()
#         # Define a convolutional layer with a large kernel to enforce smoothness over a larger spatial region
#         # The kernel size and other parameters would be chosen based on the specific requirements of the task
#         self.smoothing_conv = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, groups=channel, bias=False)
#         self.sigmoid = nn.Sigmoid()  # Sigmoid to scale the phase adjustments softly

#         # Initialize the convolutional weights to act as a smoothing kernel (like a Gaussian blur)
#         nn.init.constant_(self.smoothing_conv.weight, 1/25)  # For a 5x5 kernel, sum should be 1

#     def forward(self, phase):
#         # Apply the smoothing convolution
#         smoothed_phase = self.smoothing_conv(phase)
#         # Apply sigmoid to scale the phase adjustments
#         scaled_phase = self.sigmoid(smoothed_phase)
#         return scaled_phase


# class FSSB(nn.Module):
#     def __init__(self, c=3, reduced_dim=None): # w=128, h=128,
#     # def __init__(self, c=3, w=128, h=128, reduced_dim=None):
#         super(FSSB, self).__init__()

#         self.reduced_dim = reduced_dim
#         self.convb = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=4, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(), # ReLU(),
#             nn.Conv2d(in_channels=4, out_channels=c, kernel_size=3, stride=1, padding=1)
#         )
#         self.complex_conv = ComplexConv2d(c, c, kernel_size=3, stride=1, padding=1)

#         # # Simplified Mamba models
#         # self.model_amp = Mamba(d_model=c, d_state=8, d_conv=4, expand=1)
#         # self.model_pha = Mamba(d_model=c, d_state=8, d_conv=4, expand=1)
#         self.model_amp = Mamba(d_model=c, d_state=16, d_conv=4, expand=1)
#         self.model_pha = Mamba(d_model=c, d_state=16, d_conv=4, expand=1)

#         # Optimized 1D convolutional layers
#         self.con1_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
#         self.con1_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

#         # Cross modulation layers adapted to reduced dimensions
#         self.cross_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
#         self.cross_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

#         # Remaining layers
#         self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
#         self.ln = nn.LayerNorm(normalized_shape=(c))
#         # self.bn = nn.BatchNorm2d(c)
#         self.softmax = nn.Softmax(dim=1)

#         self.amplitude_scaling = DynamicAmplitudeScaling(c)
#         self.phase_continuity = PhaseContinuityLayer(c)

#     def forward(self, x):
#         # Compute FFT to get amplitude and phase
#         fft_x = torch.fft.fft2(x)
#         amp = torch.real(fft_x)
#         pha = torch.imag(fft_x)

#         # Apply Dynamic Amplitude Scaling and Phase Continuity
#         amp_scaled = self.amplitude_scaling(amp)
#         pha_continuous = self.phase_continuity(pha)

#         # Processing with complex convolution
#         complex_input = torch.complex(amp_scaled, pha_continuous)
#         complex_processed = self.complex_conv(complex_input)

#         # Separate processed amplitude and phase
#         processed_amp = torch.real(complex_processed)
#         processed_pha = torch.imag(complex_processed)

#         # # Initial convolutions
#         # processed_amp = self.convb(amp_scaled) + amp
#         # processed_pha = self.convb(pha_continuous) + pha

#         # Flatten for layer normalization and 1D convolutions
#         b, c, w, h = processed_amp.shape
#         flat_dim = w * h
#         processed_amp_flat = processed_amp.view(b, c, flat_dim).permute(0, 2, 1)
#         processed_pha_flat = processed_pha.view(b, c, flat_dim).permute(0, 2, 1)

#         # amp_ln = self.bn(processed_amp_flat.reshape(b, c, w, h)) # self.ln(processed_amp_flat)
#         # pha_ln = self.bn(processed_pha_flat.reshape(b, c, w, h)) # self.ln(processed_pha_flat)
#         # amp_ln, pha_ln = amp_ln.view(b, c, flat_dim).permute(0, 2, 1), pha_ln.view(b, c, flat_dim).permute(0, 2, 1)
#         amp_ln = self.ln(processed_amp_flat)
#         pha_ln = self.ln(processed_pha_flat)

#         # Process amplitude and phase with Mamba models
#         amp_mamba = self.model_amp(amp_ln)
#         pha_mamba = self.model_pha(pha_ln)

#         # Apply 1D convolutions
#         amp_conv = self.con1_1(amp_mamba.view(b, self.reduced_dim, -1))
#         pha_conv = self.con1_2(pha_mamba.view(b, self.reduced_dim, -1))

#         # Apply softmax for normalization
#         amp_softmax = self.softmax(amp_conv)
#         pha_softmax = self.softmax(pha_conv)

#         # Apply cross modulation
#         attention_amp = self.cross_1(amp_softmax) * self.cross_2(pha_softmax)
#         attention_pha = self.cross_1(pha_softmax) * self.cross_2(amp_softmax)

#         # Reshape back to original dimensions and apply attention
#         amp_out = amp_conv.view(b, c, w, h) + attention_amp.view(b, c, w, h)
#         pha_out = pha_conv.view(b, c, w, h) + attention_pha.view(b, c, w, h)

#         # Combine amplitude and phase, and apply inverse FFT
#         # output_fft = torch.complex(amp_out * torch.cos(pha_out), amp_out * torch.sin(pha_out))
#         output_fft = torch.polar(amp_out, pha_out)
#         output = torch.fft.ifft2(output_fft).real

#         # Apply final smoothing convolution
#         output = self.smooth(output)
#         return output


# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.ub = FSSB(c=in_channels, reduced_dim=32) # w=128, h=128,
#         # self.ub = FSSB(c=in_channels, reduced_dim=256) # w=128, h=128,

#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             # nn.BatchNorm2d(mid_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_channels),
#             # nn.ReLU(inplace=True)
#             DepthwiseSeparableConv(in_channels, mid_channels),
#             nn.BatchNorm2d(mid_channels),
#             nn.GELU(), # nn.ReLU(inplace=True),
#             DepthwiseSeparableConv(mid_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(), # nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         inputs = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=True)
#         outputs = self.ub(inputs)
#         outputs = F.interpolate(outputs, size=[x.shape[2], x.shape[3]],   mode='bilinear', align_corners=True) + x
#         return self.double_conv(outputs)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv and concatenate an additional input tensor"""
    
#     def __init__(self, in_channels, out_channels, merge_mode='concat'):
#         super(Down, self).__init__()
#         self.merge_mode = merge_mode
        
#         # The DoubleConv module might need adjustment based on the merge_mode
#         if merge_mode == 'concat':
#             # When concatenating, the number of input channels to the second convolution doubles
#             self.maxpool_conv = nn.Sequential(
#                 nn.MaxPool2d(2),
#                 DoubleConv(in_channels, out_channels, in_channels * 2)
#             )
#         else:
#             # No concatenation; keep as is
#             self.maxpool_conv = nn.Sequential(
#                 nn.MaxPool2d(2),
#                 DoubleConv(in_channels, out_channels)
#             )

#     def forward(self, x1, x2=None):
#         x1 = self.maxpool_conv(x1)

#         if self.merge_mode == 'concat' and x2 is not None:
#             # Ensuring dimensionality match for concatenation
#             diffY = x2.size()[2] - x1.size()[2]
#             diffX = x2.size()[3] - x1.size()[3]
#             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                             diffY // 2, diffY - diffY // 2])
            
#             # Concatenate along the channel dimension
#             x = torch.cat([x2, x1], dim=1)
#             return x
#         else:
#             return x1


# class Up(nn.Module):
#     """
#     Upscaling then double conv
#     """
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class ExpoMamba(nn.Module):
#     """ 
#     Full assembly of the parts to form the complete network 
#     -------
#     The proposed network consists of two CNNs: the first part is to use the first-order 
#     unfolding Taylor’s formula to build an interpretable network, and combine two UNets 
#     in the form of first-order Taylor’s polynomials. Then we use this constructed network 
#     to extract the feature maps of the low-resolution input image, and finally process 
#     the feature maps to form a multi-dimensional tensor termed a bilateral grid that acts 
#     on the original image to yield an enhanced result. The second part is the image 
#     enhancement using the bilateral grid. In addition, we propose a polynomial channel 
#     enhancement method to enhance UHD images.
#     """
#     def __init__(self, n_channels, bilinear=True, base_channels=32):
#         super(ExpoMamba, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear

#         factor = 2 if bilinear else 1
#         # Initial convolution block
#         self.inc = DoubleConv(n_channels, base_channels)
#         # Downsampling path
#         self.down1 = Down(base_channels, base_channels * 2)
#         self.down2 = Down(base_channels * 2, base_channels * 4)
#         self.down3 = Down(base_channels * 4, base_channels * 8)
#         self.down4 = Down(base_channels * 8, (base_channels * 16//factor))

#         # Mamba blocks
#         # self.early_mamba = Mamba(d_model=base_channels, d_state=16, d_conv=4, expand=1)
#         # self.mamba_bottleneck = Mamba(d_model=((base_channels*16)//factor), d_state=4, d_conv=2, expand=1)

#         # Upsampling path
#         self.up1 = Up( (base_channels * 16), (base_channels * 8) // factor, bilinear)
#         self.up2 = Up((base_channels * 8), (base_channels * 4) // factor, bilinear)
#         self.up3 = Up((base_channels * 4), (base_channels * 2) // factor, bilinear)
#         self.up4 = Up((base_channels * 2), base_channels, bilinear)
#         self.outc = OutConv(base_channels, n_channels)

#     def forward(self, inp):
#         x = inp
#         x1 = self.inc(x)

#         # b, c, h, w = x1.size()
#         # x1 = x1.view(b, c, h*w).permute(0, 2, 1)
#         # x1 = self.early_mamba(x1)
#         # x1 = x1.permute(0, 2, 1).view(b, c, h, w)

#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         # b, c, h, w = x5.size()
#         # x5_flattened = x5.view(b, c, h * w).permute(0, 2, 1)
#         # x5_mamba = self.mamba_bottleneck(x5_flattened)
#         # x5_mamba = x5_mamba.permute(0, 2, 1).view(b, c, h, w)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x) + inp
#         return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    model = ExpoMamba(n_channels=3, bilinear=True, base_channels=32).to(device)
    torchinfo.summary(model, input_size=(batch_size, 3, 128, 128))

    import time
    model.eval()
    # Create a dummy input tensor of FHD resolution (3x1080x1920)
    # and move it to the same device as the model
    dummy_input = torch.rand(1, 3, 128, 128).to(device)

    # Warm-up runs for more accurate timing, especially for GPU inference
    for _ in range(5):
        _ = model(dummy_input)

    # torch.cuda.synchronize()  # Wait for all kernels to finish (GPU only)
    start_time = time.time()

    # Measure inference
    _ = model(dummy_input)

    # torch.cuda.synchronize()  # Wait for all kernels to finish (GPU only)
    end_time = time.time()

    # Calculate and print the inference time
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.6f} seconds")








    #     super(FSSB, self).__init__()

    #     self.reduced_dim = reduced_dim
    #     self.convb = nn.Sequential(
    #         nn.Conv2d(in_channels=c, out_channels=4, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(in_channels=4, out_channels=c, kernel_size=3, stride=1, padding=1)
    #     )

    #     # Simplified Mamba models
    #     self.model_amp = Mamba(d_model=c, d_state=8, d_conv=4, expand=1)
    #     self.model_pha = Mamba(d_model=c, d_state=8, d_conv=4, expand=1)

    #     # Optimized 1D convolutional layers
    #     self.con1_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
    #     self.con1_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

    #     # Cross modulation layers adapted to reduced dimensions
    #     self.cross_1 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)
    #     self.cross_2 = nn.Conv1d(in_channels=self.reduced_dim, out_channels=self.reduced_dim, kernel_size=3, padding=1)

    #     # Remaining layers
    #     self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
    #     self.ln = nn.LayerNorm(normalized_shape=(c))
    #     self.bn = nn.BatchNorm2d(c)
    #     self.softmax = nn.Softmax(dim=1)

    # def forward(self, x):
    #     # Compute FFT to get amplitude and phase
    #     fft_x = torch.fft.fft2(x)
    #     amp = torch.real(fft_x)
    #     pha = torch.imag(fft_x)

    #     b, c, w, h = amp.shape
    #     flat_dim = w * h

    #     # Initial convolutions
    #     processed_amp = amp + self.convb(amp)
    #     processed_pha = pha + self.convb(pha)
    #     processed_amp = self.ln(processed_amp.reshape(b, c, -1))
    #     processed_pha = self.ln(processed_pha.reshape(b, c, -1))

    #     processed_amp = self.model_amp(processed_amp)
    #     processed_amp_1_1 = self.con1_1(processed_amp)
    #     processed_amp_sm = self.sm(processed_amp)
    #     processed_amp = torch.mul(processed_amp_1_1, processed_amp_sm)
    #     processed_amp_output = processed_amp.reshape(b,c,w,h)

    #     processed_pha = self.model_pha(processed_pha)
    #     processed_pha_1_2 = self.con1_2(processed_pha)
    #     processed_pha_sm = self.sm(processed_pha)
    #     processed_pha = torch.mul(processed_pha_1_2, processed_pha_sm)
    #     processed_pha_output = processed_pha.reshape(b,c,w,h)

    #     processed_amp_2 = self.cross_1(processed_amp_sm)
    #     processed_amp_3 = self.sm(processed_amp_2)

    #     processed_pha_2 = self.cross_1(processed_pha_sm)
    #     processed_pha_3 = self.sm(processed_pha_2)

    #     amp_modulation_map = torch.mul(processed_amp_2, processed_pha_3)
    #     amplitude = amp + processed_amp_output + amp_modulation_map.reshape(b,c,w,h)

    #     pha_modulation_map = torch.mul(processed_amp_3, processed_pha_2)
    #     phase = pha + processed_pha_output + pha_modulation_map.reshape(b,c,w,h)

    #     # Combine amplitude and phase, and apply inverse FFT
    #     output_fft = torch.real(torch.fft.ifft2(torch.complex(amplitude, phase)))
    #     return output_fft