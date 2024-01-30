import torch
import torch.nn as nn
import typing
import math

# ------------------ helper functions ------------------

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

# ------------------ UNET ------------------
    
class UnetBasicEncoderBlock(nn.Module):

    def __init__(self, in_c, c, n, stride=1, padding=0, dropout_rate=0, verbose=False):
        """
        in_c : input channel
        c : inner channel
        n : inner channel multiplier
        padding : original is 0 from paper
        """
        super(UnetBasicEncoderBlock, self).__init__()
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv2d(in_c, c*n, 3, stride, padding),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(c*n, c*n, 3, stride, padding),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x1 = self.block(x)
        x = self.maxpool(x1)
        if self.verbose is True:
            print('x, x1 \t\t:', x.shape, x1.shape)
        return x, x1

class UnetBasicDecoderBlock(nn.Module):
    def __init__(self, in_c, c, n=2, stride=1, padding=0, transpose=True, dropout_rate=0, verbose=False):
        super(UnetBasicDecoderBlock, self).__init__()
        self.transpose = transpose
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv2d(in_c, c * n, 3, stride, padding),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * n, c * n, 3, stride, padding),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(inplace=True),
        )
        # here output channel is half to concat to residual connections
        self.transpose2d = nn.ConvTranspose2d(c * n, c, 2, 2, output_padding=0)

    def forward(self, x):
        x = self.block(x)
        if self.verbose is True:
            print('x untranspose \t:', x.shape)
        if self.transpose:
            x = self.transpose2d(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x


class UnetFinalLayer(nn.Module):

    def __init__(self, in_c, out_c, kernel=1, verbose=False):
        super(UnetFinalLayer, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel)

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x



class Unet(nn.Module):
    def __init__(self, in_c=1, out_c=2, c=32, n_latent=2, expand_decoder=False, padding=0, dropout_rate=0, verbose=False):
        """
        A U-Net implementation in Pytorch.
        [*] Default value will output tensor shape (2, 388, 388) from input shape (1, 572, 572).
        [*] If image dimension is (x, 256, 256), use padding=1
        [*] Smallest input dimension possible = (16, 16)

        Arguments:
            - in_c (int) : input channel
            - c (int) : inner channel of the convolutional layer
            - n_latent (int) : latent channel multiplier, by default n_latent=2 give 1024 latent channels
            - expand_decoder (bool): option to either expand the decoder according to n_latent 
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
        self.layer1 = UnetBasicEncoderBlock(in_c, c, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
        self.layer2 = UnetBasicEncoderBlock(c*n**1, c*n**1, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
        self.layer3 = UnetBasicEncoderBlock(c*n**2, c*n**2, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
        self.layer4 = UnetBasicEncoderBlock(c*n**3, c*n**3, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)

        # latent space
        self.layer5 = UnetBasicDecoderBlock(c*n**4, c*n**4, n_latent, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)

        # decoder
        
        if expand_decoder==False:
            self.layer6 = UnetBasicDecoderBlock(c*n**5, c*n**3, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer7 = UnetBasicDecoderBlock(c*n**4, c*n**2, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer8 = UnetBasicDecoderBlock(c*n**3, c*n**1, n, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer9 = UnetBasicDecoderBlock(c*n**2, c, n, padding=padding, transpose=False, dropout_rate=dropout_rate, verbose=self.verbose)
            # final layer
            self.layer10 = UnetFinalLayer(c*n, out_c, kernel=1, verbose=self.verbose)

        elif expand_decoder==True:
            self.layer6 = UnetBasicDecoderBlock(c*n**5, c*n**3, n_latent, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer7 = UnetBasicDecoderBlock(c*n**4, c*n**2, n_latent, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer8 = UnetBasicDecoderBlock(c*n**3, c*n**1, n_latent, padding=padding, dropout_rate=dropout_rate, verbose=self.verbose)
            self.layer9 = UnetBasicDecoderBlock(c*n**2, c, n_latent, padding=padding, transpose=False, dropout_rate=dropout_rate, verbose=self.verbose)
            # final layer
            self.layer10 = UnetFinalLayer(c*n_latent, out_c, kernel=1, verbose=self.verbose)


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


# ------------------ UNET 3D ------------------


class Unet3DBasicEncoderBlock(nn.Module):
    def __init__(self, in_c, c, n, stride=1, padding=0, verbose=False):
        super(Unet3DBasicEncoderBlock, self).__init__()
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv3d(in_c, c*n, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(c*n, c*n, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.block(x)
        x = self.maxpool(x1)
        if self.verbose is True:
            print('x, x1 \t\t:', x.shape, x1.shape)
        return x, x1

class Unet3DBasicDecoderBlock(nn.Module):
    def __init__(self, in_c, c, n=2, stride=1, padding=0, transpose=True, verbose=False):
        super(Unet3DBasicDecoderBlock, self).__init__()
        self.transpose = transpose
        self.verbose = verbose
        self.block = nn.Sequential(
            nn.Conv3d(in_c, c * n, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(c * n, c * n, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.transpose3d = nn.ConvTranspose3d(c * n, c, kernel_size=2, stride=2, output_padding=0)

    def forward(self, x):
        x = self.block(x)
        if self.verbose is True:
            print('x untranspose \t:', x.shape)
        if self.transpose:
            x = self.transpose3d(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x

class Unet3DFinalLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel=1, verbose=False):
        super(Unet3DFinalLayer, self).__init__()
        self.verbose = verbose
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=kernel)

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose is True:
            print('x \t\t:', x.shape)
        return x

class Unet3D(nn.Module):
    def __init__(self, in_c=1, out_c=2, c=32, n_latent=2, expand_decoder=False, padding=0, verbose=False):
        """
        Update version of Unet with 3D convolutions.
        [*] Default value will output tensor shape (2, 388, 388) from input shape (1, 572, 572). -not check
        [*] If image dimension is (x, 256, 256), use padding=1 -not check
        [*] Smallest input dimension possible = (16, 16) -not check

        Arguments:
            - in_c (int) : input channel
            - c (int) : inner channel of the convolutional layer
            - n_latent (int) : latent channel multiplier, by default n_latent=2 give 1024 latent channels
            - expand_decoder (bool): option to either expand the decoder according to n_latent 
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
        super(Unet3D, self).__init__()

        self.verbose = verbose
        self.padding = padding
        n = 2  # inner channel multiplier

        # encoder
        self.layer1 = Unet3DBasicEncoderBlock(in_c, c, n, padding=padding, verbose=self.verbose)
        self.layer2 = Unet3DBasicEncoderBlock(c * n**1, c * n**1, n, padding=padding, verbose=self.verbose)
        self.layer3 = Unet3DBasicEncoderBlock(c * n**2, c * n**2, n, padding=padding, verbose=self.verbose)
        self.layer4 = Unet3DBasicEncoderBlock(c * n**3, c * n**3, n, padding=padding, verbose=self.verbose)

        # latent space
        self.layer5 = Unet3DBasicDecoderBlock(c * n**4, c * n**4, n_latent, padding=padding, verbose=self.verbose)

        # decoder
        if not expand_decoder:
            self.layer6 = Unet3DBasicDecoderBlock(c * n**5, c * n**3, n, padding=padding, verbose=self.verbose)
            self.layer7 = Unet3DBasicDecoderBlock(c * n**4, c * n**2, n, padding=padding, verbose=self.verbose)
            self.layer8 = Unet3DBasicDecoderBlock(c * n**3, c * n**1, n, padding=padding, verbose=self.verbose)
            self.layer9 = Unet3DBasicDecoderBlock(c * n**2, c, n, padding=padding, transpose=False, verbose=self.verbose)
            # final layer
            self.layer10 = Unet3DFinalLayer(c * n, out_c, kernel=1, verbose=self.verbose)
        else:
            self.layer6 = Unet3DBasicDecoderBlock(c * n**5, c * n**3, n_latent, padding=padding, verbose=self.verbose)
            self.layer7 = Unet3DBasicDecoderBlock(c * n**4, c * n**2, n_latent, padding=padding, verbose=self.verbose)
            self.layer8 = Unet3DBasicDecoderBlock(c * n**3, c * n**1, n_latent, padding=padding, verbose=self.verbose)
            self.layer9 = Unet3DBasicDecoderBlock(c * n**2, c, n_latent, padding=padding, transpose=False, verbose=self.verbose)
            # final layer
            self.layer10 = Unet3DFinalLayer(c * n_latent, out_c, kernel=1, verbose=self.verbose)

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

