import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from .op import upfirdn2d, conv2d_gradfix
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


"""Modulation convolution"""
class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        demodulate=True,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self.demodulate = demodulate
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.modulation = nn.Linear(512, in_channel)


        # si in Formula 1 of the paper
        fan_in = in_channel * kernel_size ** 2  # 4608
        self.scale = 1 / math.sqrt(fan_in)  # 0.014

        # Modulation and demodulation of convolutional kernels
        self.weight = nn.Parameter(  # out_channel=512 in_channel=512 kernel_size=3 kernel_size=3
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        if upsample:
            factor = 2
            p = (4 - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur([1, 3, 3, 1], pad=(pad0, pad1), upsample_factor=factor)



    def forward(self, input, style):
        batch, in_channel, height, width = input.shape # 1 512 4 4
        # For weight modulation, style is latentcode
        style = self.modulation(style) # (1,512)
        style = style.view(batch, 1, in_channel, 1, 1) # style (1,512) -> (1,1,512,1,1)
        weight = self.scale * self.weight * style # Corresponding paper formula 1, (1,512,512,3,3) modulation
        # demodulate
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(  # (512,512,3,3)
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        # Up-sampling
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width) # [1,512,4,4]
            weight = weight.view( # [1,512,512,3,3]
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape( # [512,512,3,3]
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d( # [1,512,9,9] deconvolution
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)# [1,512,9,9]
            out = self.blur(out)# [1,512,8,8]
        # Maintain original size
        else:
            input = input.view(1, batch * in_channel, height, width) # [1,512,4,4]
            # Using Convolutional Pairs after Modulation and Demodulation for const_ input for convolution
            out = conv2d_gradfix.conv2d(  # [1,512,4,4]
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width) # [1,512,4,4]
        return out
"""Add noise"""
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None): # image [1，512，4，4]
        if noise is None:
            batch, _, height, width = image.shape # 1 512 4 4
            noise = image.new_empty(batch, 1, height, width).normal_() # Returns a tensor of size filled with uninitialized data. [1,1,4,4]

        return image + self.weight * noise


"""Style Convolutional Module"""
class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel, # 512
        out_channel, # 512
        kernel_size, # 3
        upsample=False, # False
        demodulate=True, #True
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,  # 512
            out_channel,  # 512
            kernel_size,  # 3
            upsample=upsample,  # False
            demodulate=demodulate,  # True
        )

        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style) # Modulation demodulation and convolution in the paper
        out = self.noise(out, noise=noise) # Adding noise in the paper
        # out = out + self.bias
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

        self.conv = ModulatedConv2d(in_channel=in_channel, out_channel=3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None): # input (1,512,4,4) style (1,512)
        out = self.conv(input, style) # (1,3,4,4)
        out = out + self.bias # self.bias (1,3,1,1) out (1,3,4,4)

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out

class Generator_StyleGAN2(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        """映射网络"""
        layers = [PixelNorm()]
        for i in range(8):
            layers.append(
                nn.Sequential(
                    nn.Linear(512, 512, bias=True),
                    nn.LeakyReLU(512, True),
                )
            )
        self.style = nn.Sequential(*layers)

        # Input trainable constant
        self.input = nn.Parameter(torch.randn(1, 512, 4, 4))


        self.conv1 = StyledConv(  # self.channels[4]=512  3 style_dim=512
            in_channel=512, out_channel=512, kernel_size=3,
        )

        self.to_rgb1 = ToRGB(512, upsample=False)

        self.log_size = int(math.log(521, 2)) # 9
        self.num_layers = (self.log_size - 2) * 2 + 1 # 15

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        """
                     register_buffer(name, tensor, persistent=True)
                     name(string) -The name of the buffer. You can access the buffer from this module with the given name.
                     tensor(Tensor or None) -The buffer to register.
                     This is typically used to register buffers that should not be considered model parameters.
        """
        self.noises.register_buffer(f"noise_{0}", torch.randn(*[1,1,4,4]))
        self.noises.register_buffer(f"noise_{1}", torch.randn(*[1,1,8,8]))
        self.noises.register_buffer(f"noise_{2}", torch.randn(*[1,1,8,8]))
        self.noises.register_buffer(f"noise_{3}", torch.randn(*[1,1,16,16]))
        self.noises.register_buffer(f"noise_{4}", torch.randn(*[1,1,16,16]))
        self.noises.register_buffer(f"noise_{5}", torch.randn(*[1,1,32,32]))
        self.noises.register_buffer(f"noise_{6}", torch.randn(*[1,1,32,32]))
        self.noises.register_buffer(f"noise_{7}", torch.randn(*[1,1,64,64]))
        self.noises.register_buffer(f"noise_{8}", torch.randn(*[1,1,64,64]))
        self.noises.register_buffer(f"noise_{9}", torch.randn(*[1,1,128,128]))
        self.noises.register_buffer(f"noise_{10}", torch.randn(*[1,1,128,128]))
        self.noises.register_buffer(f"noise_{11}", torch.randn(*[1,1,256,256]))
        self.noises.register_buffer(f"noise_{12}", torch.randn(*[1,1,256,256]))
        self.noises.register_buffer(f"noise_{13}", torch.randn(*[1,1,512,512]))
        self.noises.register_buffer(f"noise_{14}", torch.randn(*[1,1,512,512]))

        """Layer 2"""
        # Style module with upsampling
        self.convs.append(
            StyledConv( # With upsampling
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv( # Without upsampling
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512)) #Convert to Image
        """Layer 3"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))
        """Layer 4"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))

        """Layer 5"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))
        """Layer 6"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=256,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                256, 256, 3,
            )
        )
        self.to_rgbs.append(ToRGB(256))
        """Layer 7"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=256,
                out_channel=128,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                128, 128, 3,
            )
        )
        self.to_rgbs.append(ToRGB(128))
        """Layer 8"""
        # Style module with upsampling
        self.convs.append(
            StyledConv(
                in_channel=128,
                out_channel=64,
                kernel_size=3,
                upsample=True,
            )
        )
        # Style module without upsampling
        self.convs.append(
            StyledConv(
                64, 64, 3,
            )
        )
        self.to_rgbs.append(ToRGB(64))

    def forward(self,styles):
        # latentcode. Perform 18 layers of Linear map
        style0 = self.style(styles[0])
        style1= self.style(styles[1])
        styles = [style0,style1] # [ (1,512) ,(1,512)]
        # noise
        noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]  #num_layers=15 [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        # Randomly reorganize two w of Linear map into a latentcode of [1,16,512] with a random number.
        inject_index = random.randint(1, 16 - 1) # (1,15) Random as inject_index. 5
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1) # [1,inject_index,512] eq. [1,5,512]
        latent2 = styles[1].unsqueeze(1).repeat(1, 16 - inject_index, 1) # [1,16 - inject_index,512] eq. [1,16-5==11,512]
        latent = torch.cat([latent, latent2], 1) # [1,16,512] merge

        const_input = self.input.repeat(latent.shape[0], 1, 1, 1) # [1,512,4,4]
        # Layer 1
        out0 = self.conv1(const_input, latent[:, 0], noise=noise[0])  # [1,512,4,4]
        skip0 = self.to_rgb1(out0, latent[:, 1])#out [1,512,4,4]  latent[:, 1] [1, 512] -> [1,3,4,4]
        # Layer 2
        out1 = self.convs[0](out0, latent[:, 1], noise=noise[1]) # [1,512,8,8] upsampling
        out1 = self.convs[1](out1, latent[:, 2], noise=noise[2]) # [1,512,8,8]
        skip1 = self.to_rgbs[0](out1, latent[:, 3], skip0)# [1,3,8,8]
        # Layer 3
        out2 = self.convs[2](out1, latent[:, 3], noise=noise[3])  # [1,512,16,16] upsampling
        out2 = self.convs[3](out2, latent[:, 4], noise=noise[4])  # [1,512,16,16]
        skip2 = self.to_rgbs[1](out2, latent[:, 5], skip1)  # [1,3,16,16]
        # Layer 4
        out3 = self.convs[4](out2, latent[:, 5], noise=noise[5])  # [1,512,32,32] upsampling
        out3 = self.convs[5](out3, latent[:, 6], noise=noise[6])  # [1,512,32,32]
        skip3 = self.to_rgbs[2](out3, latent[:, 7], skip2)  # [1,3,32,32]
        # Layer 5
        out4 = self.convs[6](out3, latent[:, 7], noise=noise[7])  # [1,512,64,64] upsampling
        out4 = self.convs[7](out4, latent[:, 8], noise=noise[8])  # [1,512,64,64]
        skip4 = self.to_rgbs[3](out4, latent[:, 9], skip3)  # [1,3,64,64]
        # Layer 6
        out5 = self.convs[8](out4, latent[:, 9], noise=noise[9])  # [1,256,128,128] upsampling
        out5 = self.convs[9](out5, latent[:, 10], noise=noise[10])  # [1,256,128,128]
        skip5 = self.to_rgbs[4](out5, latent[:, 11], skip4)  # [1,3,128,128]
        # Layer 7
        out6 = self.convs[10](out5, latent[:, 11], noise=noise[11])  # [1,128,256,256] upsampling
        out6 = self.convs[11](out6, latent[:, 12], noise=noise[12])  # [1,128,256,256]
        skip6 = self.to_rgbs[5](out6, latent[:, 13], skip5)  # [1,3,256,256]
        # Layer 8
        out7 = self.convs[12](out6, latent[:, 13], noise=noise[13])  # [1,64,512,521] upsampling
        out7 = self.convs[13](out7, latent[:, 14], noise=noise[14])  #  [1,64,512,521]
        skip7 = self.to_rgbs[6](out7, latent[:, 15], skip6)  # [1,3,512,512]
        image = skip7

        # return image, latent
        return image