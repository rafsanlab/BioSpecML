import torch
import torch.nn as nn
import typing
import math

# ------------------ SIMPLE CONVOLUTIONS ------------------

class Conv2DNet(nn.Module):
    """ A simple convolution 2D network.
        Example use:
            tensor = torch.rand(1, 240, 256, 256) # B, C, W, H
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


class Conv3DNet(nn.Module):
    """ A simple convolution 2D network.
        Example use:
            tensor = torch.rand(1, 1, 240, 256, 256) # B, C, D, W, H
            model = Conv2DNet(input_channels=1, temporal_dim=240, channel_multiplier=1, num_classes=10, width=256, height=256)
    """
    def __init__(self, input_channels, temporal_dim, channel_multiplier, num_classes, width, height):
        super(Conv3DNet, self).__init__()
        in_c = input_channels
        out_c = in_c*channel_multiplier*2

        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.norm1 = nn.BatchNorm3d(out_c)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(out_c, out_c*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.norm2 = nn.BatchNorm3d(out_c*2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(out_c*2, out_c*4, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.norm3 = nn.BatchNorm3d(out_c*4)
        self.relu3 = nn.ReLU()

        in_fc = (out_c*4) * (math.ceil(temporal_dim/(2*4))) * (math.ceil(width/(2*4))) * (math.ceil(height/(2*4)))
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


# ------------------ RESNET ------------------

#--- basic block ---

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:

        super(BasicBlock, self).__init__()
        """ <expansion> is a multiplicative factor for the subsequent
            conv2d layer's output channels. It is 1 for ResNet18 and ResNet34.
        """
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

#--- building resnet ---

class ResNet(nn.Module):
    def __init__(self, img_channels:int, num_layers:int,
                 block:typing.Type[BasicBlock], num_classes:int=1000) -> None:

        super(ResNet, self).__init__()

        if num_layers == 18:
            """ <layers> is a list that defines the number of `BasicBlock`
                to use to build the network and how many basic blocks to stack
            """
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 256 ## 64
        """ All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the
            first three layers. Here, <kernel_size> is 7.
        """
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0]) ## 64
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2) ## 128
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2) ## 256
        self.layer4 = self._make_layer(block, 2048, layers[3], stride=2) ## 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048*self.expansion, num_classes) ## 512

    def _make_layer(self, block:typing.Type[BasicBlock], out_channels:int,
        blocks:int, stride:int=1) -> nn.Sequential:

        downsample = None
        if stride != 1:
            """ This should pass from `layer2` to `layer4` or
                when building ResNets50 and above. Section 3.3 of the paper
                Deep Residual Learning for Image Recognition
                (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        """ The spatial dimension of the final layer's feature
            map should be (7, 7) for all ResNets.
        """
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
# ------------------ UNET ------------------

def crop_and_concat(src_tnsr, tgt_tnsr, concat=True, verbose=False):

    error_dim = 'Tensor dimension not the same, expect both width and height to be same.'

    # check if image is batched or not
    if src_tnsr.dim() == 3:
        if src_tnsr.size()[1] != src_tnsr.size()[2]:
            raise Exception(error_dim)
        if tgt_tnsr.size()[1] != tgt_tnsr.size()[2]:
            raise Exception(error_dim)

        # get crop param
        src_size = src_tnsr.size()[1] # get width
        tgt_size = tgt_tnsr.size()[1]
        delta = src_size - tgt_size
        delta = delta // 2

        # crop
        crop_tnsr = src_tnsr[:, delta:src_size-delta, delta:src_size-delta]

    # if tensor is batched
    elif src_tnsr.dim() == 4:
        if src_tnsr.size()[2] != src_tnsr.size()[3]:
            raise Exception(error_dim)
        if tgt_tnsr.size()[2] != tgt_tnsr.size()[3]:
            raise Exception(error_dim)

        # get crop param
        src_size = src_tnsr.size()[2] # get width
        tgt_size = tgt_tnsr.size()[2]
        delta = src_size - tgt_size
        delta = delta // 2

        # crop
        crop_tnsr = src_tnsr[:, :, delta:src_size-delta, delta:src_size-delta]

    else:
        print('Tensor shape not supported, try (3, 256, 256) or (1, 3, 256, 256)')

    # print output shape
    if verbose:
        print(crop_tnsr.shape, tgt_tnsr.shape)

    # concat tensor
    if concat:
        if src_tnsr.dim() == 3:
            concat_tnsr = torch.cat([crop_tnsr, tgt_tnsr], dim=0)
            return concat_tnsr
        if src_tnsr.dim() == 4:
            concat_tnsr = torch.cat([crop_tnsr, tgt_tnsr], dim=1)
            return concat_tnsr
    else:
        return concat_tnsr


class BasicEncoderBlock(nn.Module):

    def __init__(self, in_c, c, n, stride=1, padding=0, verbose=False):
        """
        in_c : input channel
        c : inner channel
        n : inner channel multiplier
        padding : original is 0 from paper
        """
        super(BasicEncoderBlock, self).__init__()
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv2d(in_c, c*n, 3, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(c*n, c*n, 3, stride, padding),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x1 = self.block(x)
        x = self.maxpool(x1)
        if self.verbose is True:
            print('x, x1 \t\t:', x.shape, x1.shape)
        return x, x1

class BasicDecoderBlock(nn.Module):
    def __init__(self, in_c, c, n=2, stride=1, padding=0, transpose=True, verbose=False):
        super(BasicDecoderBlock, self).__init__()
        self.transpose = transpose
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv2d(in_c, c * n, 3, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * n, c * n, 3, stride, padding),
            nn.ReLU(inplace=True),
        )
        # here output channel is half to concat to residual connections
        self.transpose2d = nn.ConvTranspose2d(c * n, c, 2, 2, output_padding=0)

    def forward(self, x):
        x = self.block(x)
        print('x untranspose \t:', x.shape)
        if self.transpose:
            x = self.transpose2d(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x


class FinalLayer(nn.Module):

    def __init__(self, in_c, out_c, kernel=1, verbose=False):
        super(FinalLayer, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel)

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x


class Unet(nn.Module):
    def __init__(self, in_c=1, out_c=2, c=32, n_latent=2, padding=0, verbose=False):
        """
        A U-Net implementation in Pytorch.
        [*] Default value will output tensor shape (2, 388, 388) from input shape (1, 572, 572).
        [*] If image dimension is (x, 256, 256), use padding=1
        [*] Smallest input dimension possible = (16, 16)

        Arguments:
            - in_c (int) : input channel
            - c (int) : inner channel of the convolutional layer
            - n_latent (int) : latent channel multiplier, by default n_latent=2 give 1024 latent channels
            - padding (int) : original is 0 from paper, but use 1 for same final image dimension

        Usage:
            Example 1:
                >>> tensor = torch.rand(3, 256, 256)
                >>> model = Unet(in_c=3, padding=1, verbose=True)
                >>> outputs = model(tensor)

            Example 2:
                >>> tensor = torch.rand(1, 572, 572)
                >>> model = Unet(verbose=True)
                >>> outputs = model(tensor)

        """
        super(Unet, self).__init__()

        self.verbose = verbose
        self.padding = padding
        n = 2 # inner channel multiploier

        # encoder
        self.layer1 = BasicEncoderBlock(in_c, c, n, padding=padding, verbose=self.verbose)
        self.layer2 = BasicEncoderBlock(c*n**1, c*n**1, n, padding=padding, verbose=self.verbose)
        self.layer3 = BasicEncoderBlock(c*n**2, c*n**2, n, padding=padding, verbose=self.verbose)
        self.layer4 = BasicEncoderBlock(c*n**3, c*n**3, n, padding=padding, verbose=self.verbose)

        # latent space
        self.layer5 = BasicDecoderBlock(c*n**4, c*n**4, n_latent, padding=padding, verbose=self.verbose)

        # decoder
        self.layer6 = BasicDecoderBlock(c*n**5, c*n**3, n, padding=padding, verbose=self.verbose)
        self.layer7 = BasicDecoderBlock(c*n**4, c*n**2, n, padding=padding, verbose=self.verbose)
        self.layer8 = BasicDecoderBlock(c*n**3, c*n**1, n, padding=padding, verbose=self.verbose)
        self.layer9 = BasicDecoderBlock(c*n**2, c, n, padding=padding, transpose=False, verbose=self.verbose)

        # final layer
        self.layer10 = FinalLayer(c*n, out_c, kernel=1, verbose=self.verbose)


    def forward(self, x):
        # encoder
        x, x1 = self.layer1(x)
        x, x2 = self.layer2(x)
        x, x3 = self.layer3(x)
        x, x4 = self.layer4(x)
        # latent space
        x = self.layer5(x)
        # decoder
        x = self.layer6(crop_and_concat(x4, x))
        x = self.layer7(crop_and_concat(x3, x))
        x = self.layer8(crop_and_concat(x2, x))
        x = self.layer9(crop_and_concat(x1, x))
        x = self.layer10(x)
        return x