import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.vmamba import VSSM
from models.comfy import Block
from models.csrnet import CSRNet

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
        # A convolutional layer with a large kernel to enforce smoothness over a larger spatial region
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


class HDRLayer(nn.Module):
    def __init__(self, threshold=0.9, scale_factor=0.1):
        super(HDRLayer, self).__init__()
        self.threshold = threshold
        self.scale_factor = scale_factor

    def forward(self, x):
        # Identify overexposed regions
        overexposed_mask = (x > self.threshold).float()
        # Apply a simple tone-mapping curve: log scaling
        tone_mapped = torch.log1p(x * self.scale_factor)
        # Blend original and tone-mapped based on mask
        output = (1 - overexposed_mask) * x + overexposed_mask * tone_mapped
        return output


class HDRLayerOut(nn.Module):
    def __init__(self, threshold=0.9):
        super(HDRLayerOut, self).__init__()
        self.threshold = threshold
        self.csr_net = CSRNet(in_nc=24, out_nc=24, base_nf=16, cond_nf=8)

    def forward(self, x):
        # Identify overexposed regions
        overexposed_mask = (x > self.threshold).float()
        # Apply a simple tone-mapping curve: log scaling
        tone_mapped = self.csr_net(x)
        # Blend original and tone-mapped based on mask
        output = (1 - overexposed_mask) * x + overexposed_mask * tone_mapped
        return output


class FSSB(nn.Module):
    def __init__(self, c=3): #, reduced_dim=None):
        super(FSSB, self).__init__()

        # self.reduced_dim = reduced_dim
        self.complex_conv = ComplexConv2d(c, c, kernel_size=3, stride=1, padding=1)

        # Mamba models for processing amplitude and phase
        self.model_amp = VSSM(
            patch_size=4,
            in_chans=c,
            num_classes=c,
            depths=[1],
            # dims=[32, 64, 128],
            # dims=[64, 128, 256],
            dims=[64, 128, 256, 512],
            # dims=[32, 64, 128, 64, 32],
            d_state=16,
            d_conv=3,
            expand=3,
            norm_layer=nn.LayerNorm
        )
        self.model_pha = VSSM(
            patch_size=4,
            in_chans=c,
            num_classes=c,
            depths=[1],
            # dims=[32, 64, 128],
            dims=[64, 128, 256, 512],
            # dims=[32, 64, 128, 64, 32],
            d_state=16,
            d_conv=3,
            expand=3,
            norm_layer=nn.LayerNorm
        )

        self.amplitude_scaling = DynamicAmplitudeScaling(c)
        self.phase_continuity = PhaseContinuityLayer(c)
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.hdr_layer = HDRLayer()

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

        # Process amplitude and phase with Mamba models
        processed_amp = self.model_amp(amp_scaled)
        processed_pha = self.model_pha(pha_continuous)

        # Combine processed amplitude and phase, and apply inverse FFT
        combined_fft = torch.complex(processed_amp, processed_pha)
        output = torch.fft.ifft2(combined_fft).real

        # Apply final smoothing convolution
        output = self.smooth(output)

        # Applying HDR processing after frequency modulation
        x = self.hdr_layer(x)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False): # , use_se=True):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU() # nn.GELU()

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU() # nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class DoubleConv(nn.Module):
    """(depthwise-convolution => [BN] => GELU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.fssb = FSSB(c=in_channels)#, reduced_dim=16)

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            DepthwiseSeparableConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True), # nn.GELU(),
        )

    def forward(self, x):
        inputs = F.interpolate(x, size=[48, 48], mode='bilinear', align_corners=True)
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
    Upscaling then double conv with enhanced feature processing.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Define the upsampling method
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            channels_after_up = in_channels // 2
            # Use DoubleConv for combining upsampled features and skip connection
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            channels_after_up = in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels)

        # Process the upsampled features with the correct number of channels
        self.process = nn.Sequential(
            Block(channels_after_up, channels_after_up),
            nn.Conv2d(channels_after_up, channels_after_up, 3, padding=1, bias=False),
            nn.ReLU() # nn.GELU()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample the input
        x1 = self.process(x1)  # Process upsampled features

        # Adjust the size for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate upsampled features with the features from the skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # Apply DoubleConv to the combined features


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


class ColorCorrectionMatrix(nn.Module):
    def __init__(self, channels=3):
        super(ColorCorrectionMatrix, self).__init__()
        # Initialize a channels x channels matrix for RGB color correction
        self.ccm = nn.Parameter(torch.eye(channels))

    def forward(self, x):
        # Apply color correction
        # Assuming 'x' is a batch of images [N, C, H, W] and 'ccm' is [C, C]
        x_corrected = torch.einsum('ij,njhw->nihw', self.ccm, x)
        return x_corrected


class ExpoMamba(nn.Module):
    """ 
    Full assembly of the parts to form the complete network 
    -------
    https://github.com/zfdong-code/MNet/blob/6ce62fa09f74b179af12e1ba72a24787947cab73/MNet_pure/PyTorch/MNet.py
    """
    def __init__(self, n_channels, bilinear=True, base_channels=32, deep_supervision=True):
        super(ExpoMamba, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        factor = 2 if bilinear else 1
        # Initialize network layers...
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.color_correction = ColorCorrectionMatrix(channels=24)
        # self.gamma_correction = GammaCorrection()

        self.hdr_layer_outc = HDRLayerOut(threshold=0.95)
        self.outc = OutConv(base_channels, n_channels)

        if self.deep_supervision:
            common_channels = 3
            # Define conv layers to unify channel dimensions
            self.align_convs = nn.ModuleList([
                nn.Conv2d(base_channels, common_channels, 1), # Assuming common_channels is your target channel size
                nn.Conv2d(base_channels, common_channels, 1),
                nn.Conv2d(base_channels*2, common_channels, 1),
                nn.Conv2d(base_channels*4, common_channels, 1),
                nn.Conv2d(base_channels*2, common_channels, 1),
                nn.Conv2d(base_channels*4, common_channels, 1),
                nn.Conv2d(base_channels*8, common_channels, 1),
            ])
            self.final_conv = nn.Conv2d(common_channels * len(self.align_convs) + n_channels, n_channels, 1)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)

        if self.deep_supervision:
            features = [x_up4, x_up3, x_up2, x_up1, x2, x3, x4]
            resized_and_aligned_features = [align_conv(F.interpolate(feat, size=inp.size()[2:], mode='bilinear', align_corners=True))
                                            for feat, align_conv in zip(features, self.align_convs)]
            combined_features = torch.cat(resized_and_aligned_features + [inp], dim=1)
            combined_features = self.hdr_layer_outc(combined_features)
            combined_features = self.color_correction(combined_features)
            final_output = self.final_conv(combined_features)
        else:
            final_output = self.outc(x_up4) + inp

        # final_output = self.gamma_correction(final_output)
        # final_output = self.color_correction(final_output)

        return final_output


import time, torchinfo
from fvcore.nn import FlopCountAnalysis

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    base_channels = 48 # 32

    model = ExpoMamba(n_channels=3, bilinear=True, base_channels=base_channels).to(device)
    torchinfo.summary(model, input_size=(batch_size, 3, 256, 256))
    model.eval()

    checkpoint = torch.load("../checkpoint/lolv1/lolv1_5_Apr_18h_37m/best_ssim.pth", map_location=device)
    model.load_state_dict(checkpoint)

    dummy_input_1 = torch.rand(1, 3, 256, 256).to(device)
    dummy_input_2 = torch.rand(1, 3, 256, 256).to(device)
    dummy_input_3 = torch.rand(1, 3, 256, 256).to(device)
    dummy_input_4 = torch.rand(1, 3, 256, 256).to(device)
    dummy_input_5 = torch.rand(1, 3, 256, 256).to(device)
    dummy_input_list = [dummy_input_1, dummy_input_2, dummy_input_3, dummy_input_4, dummy_input_5]

    # Warm-up runs for more accurate timing
    for _ in range(5):
        _ = model(dummy_input_1)

    start_time = time.time()
    for dummy_input in dummy_input_list:
        output = model(dummy_input)
    end_time = time.time()

    # Calculate and print the inference time
    inference_time = (end_time - start_time)/len(dummy_input_list)
    print(f"Output shape: {output.shape}")
    print(f"Inference Time: {inference_time:.6f} seconds")

    input_ = torch.randn(1, 3, 600, 400).to(device)
    torch.cuda.synchronize()
    model.eval()
    time_start = time.time()
    _ = model(input_)
    time_end = time.time()
    torch.cuda.synchronize()

    flop_analyzer = FlopCountAnalysis(model, input_)
    total_flops = flop_analyzer.total()
    gflops = total_flops / 10**9
    print(f"GFLOPs: {gflops}")