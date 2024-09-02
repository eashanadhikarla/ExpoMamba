import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# import vgg_loss
# from torchvision.models import vgg16, VGG16_Weights
from IQA_pytorch import SSIM, LPIPSvgg
# from metrics import SSIM, PSNR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################
# Lab Color Space Loss
######################
def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]
    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b
    out: torch.Tensor = torch.stack([x, y, z], -3)
    return out

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    # Convert from Linear RGB to sRGB
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    rs: torch.Tensor = torch.where(r > 0.04045, torch.pow(((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045, torch.pow(((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045, torch.pow(((b + 0.055) / 1.055), 2.4), b / 12.92)

    image_s = torch.stack([rs, gs, bs], dim=-3)
    xyz_im: torch.Tensor = rgb_to_xyz(image_s)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1., 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    power = torch.pow(xyz_normalized, 1 / 3)
    scale = 7.787 * xyz_normalized + 4. / 29.
    xyz_int = torch.where(xyz_normalized > 0.008856, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]
    L: torch.Tensor = (116. * y) - 16.
    a: torch.Tensor = 500. * (x - y)
    _b: torch.Tensor = 200. * (y - z)
    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)
    return out

class LABLoss(nn.Module):
    def __init__(self):
        super(LABLoss, self).__init__()
    
    def forward(self, enhanced_image, original_image):
        print(enhanced_image)
        lab_output = rgb_to_lab(enhanced_image)
        lab_target = rgb_to_lab(original_image)
        return F.mse_loss(lab_output, lab_target)

##########
# YUV Loss
##########
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b
    out: torch.Tensor = torch.stack([y, u, v], -3)
    return out

class YUVLoss(nn.Module):
    def __init__(self):
        super(YUVLoss, self).__init__()
    
    def forward(self, enhanced_image, original_image):
        lab_output = rgb_to_yuv(enhanced_image)
        lab_target = rgb_to_yuv(original_image)
        return F.mse_loss(lab_output, lab_target)

##########
# HSV Loss
##########
def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

class HSVLoss(nn.Module):
    def __init__(self):
        super(HSVLoss, self).__init__()
    
    def forward(self, enhanced_image, original_image):
        hsv_output = rgb2hsv_torch(enhanced_image)
        hsv_target = rgb2hsv_torch(original_image)
        return F.mse_loss(hsv_output, hsv_target)

######################
# Color Constancy Loss
######################
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


#################
# Perceptual Loss
#################
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.mse_loss = nn.MSELoss()
        self.layer_name_mapping = {
            '3' : "relu1_2",
            '8' : "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(self.mse_loss(pred_im_feature, gt_feature))
        return sum(loss)/len(loss)

# ##########
# # VGG Loss
# ##########
# class VGG_Loss(nn.Module):
#     def __init__(self, _lambda_=0.2):
#         super(VGG_Loss, self).__init__()
#         vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
#         vgg_model = vgg_model.to(device)
#         for param in vgg_model.parameters():
#             param.requires_grad = False
#         self.loss_network = LossNetwork(vgg_model)
#         self._lambda_ = _lambda_

#     def forward(self, output, target):
#         Lvgg = self.loss_network(output, target)
#         minimizedLvgg = self._lambda_ * Lvgg
#         return minimizedLvgg

#########################
# Gradient Histogram Loss
#########################
class GradientLoss(nn.Module):
    """Gradient Histogram Loss"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2

        assert(self.clip_radius>0 and self.clip_radius<=1)
        self.bin_width = 2*self.clip_radius/self.bin_num

        if self.bin_width*255<1:
            raise RuntimeError("bin width is too small")

        self.bin_mean = np.arange(-self.clip_radius+self.bin_width*0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L1' # 'L2'

        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]     # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]      # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]    # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src
        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None

        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])
            fx = torch.cuda.FloatTensor([fx])
            fy = torch.cuda.FloatTensor([fy])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)
        loss = self.criterion(output_gradient_x,target_gradient_x)+self.criterion(output_gradient_y,target_gradient_y)
        return loss

##################
# Charbonnier Loss
##################
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)

####################################################################
# Handling HDR related issue of overexposed content with regularizer
####################################################################
class OverexposedRegularization(torch.nn.Module):
    def __init__(self, lambda_overexposed_weight=0.8):
        super(OverexposedRegularization, self).__init__()
        self.lambda_overexposed_weight = lambda_overexposed_weight

    def forward(self, base_loss, input, target):
        # Detect overexposed areas in target
        overexposed_mask = (target > 0.9).float()
        # Calculate loss specifically for overexposed areas
        overexposed_loss = torch.mean(overexposed_mask * (input - target) ** 2)
        # Total loss
        total_loss = base_loss + self.lambda_overexposed_weight * overexposed_loss
        return total_loss

class LossFunctions:
    def __init__(self):
        # L1 Loss
        self.l1_loss = nn.L1Loss()

        # Smooth-L1 Loss
        self.smooth_l1_loss = F.smooth_l1_loss

        # MSE Loss
        self.mse_loss = nn.MSELoss()

        # SSIM Loss
        # self.ssim_loss = piqa.SSIM(channels=3).to(device=device)
        # self.ssim_loss = pyiqa.create_metric('ssim', device=device, as_loss=True)
        self.ssim_loss = SSIM(channels=3)

        # LPIPS Loss
        # self.lpips_loss = piqa.LPIPS(network='vgg').to(device=device) 
        # self.lpips_loss = pyiqa.create_metric('lpips', device=device, as_loss=True)
        # self.lpips_loss = pyiqa.create_metric('lpips-vgg', device=device, as_loss=True)
        self.lpips_loss = LPIPSvgg(channels=3).to(device)

        # PSNR Metrics
        # self.psnr = pyiqa.create_metric('psnr', device=device, as_loss=False)

        # Gradient Histogram Loss
        self.gradient_hist_loss = GradientLoss()

        # Charbonnier Loss
        self.charbonnier_loss = CharbonnierLoss()

        # Color Loss
        self.color = L_color()

        # Lab Color Space Loss
        self.lab_color_loss = LABLoss()

        # YUV loss
        self.yuv_loss = YUVLoss()

        # HSV loss
        self.hsv_loss = HSVLoss()

        # Regularizer
        self.overexposed_regularizer = OverexposedRegularization(lambda_overexposed_weight=0.8)

        # VGG Loss
        # vgg_model = vgg16().features[:16]
        # vgg_model = vgg_model.to(device)
        # for param in vgg_model.parameters():
        #     param.requires_grad = False
        # self.vgg_loss = VGG_Loss()
        # self.vgg_loss.eval()