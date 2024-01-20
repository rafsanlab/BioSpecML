import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from math import exp


class SSIMLoss(nn.Module):
    """
    Adapted from https://github.com/NRauschmayr/SSIM_Loss/blob/master/ssim_loss.py
    originaly code in MXNet, converted to Pytorch.

    Example of use: in your training loop;
        >>> inputs = torch.rand(1, 3, 16, 16) # RGB with 3 channels
        >>> reference = torch.rand(1, 3, 16, 16)
        >>> criterion = SSIMLoss(size=11, channel=3) 
        >>> loss = criterion(inputs, reference)
        >>> print(f"SSIM Loss: {loss.item()}")

    """
    def __init__(self, size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.size = size
        self.channel = channel
        self.window = self.create_window(size, channel).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def create_window(self, size, channel):
        gauss = torch.Tensor([exp(-(x - size//2)**2/float(2*1.5**2)) for x in range(size)])
        window_1d = gauss / gauss.sum()
        window_2d = window_1d.view(-1, 1) * window_1d
        window = window_2d.expand(channel, 1, size, size).contiguous()
        window = window.view(channel, 1, size, size)
        window = window.requires_grad_(False)
        return window

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.size//2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()