import torch
import torch.nn as nn
import math

class Conv2DNet(nn.Module):
    """ A simple convolution 2D network.
        Example use:
            tensor = torch.rand(1, 240, 256, 256) # B, C, D, W, H
            model = Conv2DNet(input_channels=240, num_classes=10, width=256, height=256)
    """
    def __init__(self, input_channels, num_classes, width, height):
        super(Conv2DNet, self).__init__()
        in_c = input_channels
        out_c = in_c*2

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_c, out_c*2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_c*2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_c*2, out_c*4, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(out_c*4)
        self.relu3 = nn.ReLU()

        in_fc = (out_c*4) * (math.ceil(width/(2*4))) * (math.ceil(height/(2*4)))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_fc, num_classes)
 
    def forward(self, x):

        x = self.relu1(self.norm1(self.conv1(x)))
        # print('After ReLU1 \t:', x.shape)

        x = self.relu2(self.norm2(self.conv2(x)))
        # print('After ReLU2 \t:', x.shape)

        x = self.relu3(self.norm3(self.conv3(x)))
        # print('After ReLU3 \t:', x.shape)
        
        x = self.flatten(x)
        # print('After flatten \t:', x.shape)
        
        x = self.fc(x)
        return x