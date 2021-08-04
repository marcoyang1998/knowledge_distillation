#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn
#from espnet.nets.pytorch_backend.wavenet import CausalConv1d

class CausalConv1d(nn.Module):
    """1D dilated causal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, groups=1):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with the shape (B, in_channels, T).

        Returns:
            Tensor: Tensor with the shape (B, out_channels, T)

        """
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, : -self.padding]
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True, causal=False):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.causal = causal
        if causal:
            kernel_size = kernel_size//2
            self.pointwise_conv1 = CausalConv1d(
                channels,
                2 * channels,
                kernel_size=1,
                bias=bias,
            )
            self.depthwise_conv = CausalConv1d(
                channels,
                channels,
                kernel_size,
                #padding=(kernel_size - 1) // 2,
                groups=channels,
                bias=bias,
            )
            self.norm = nn.BatchNorm1d(channels)
            self.pointwise_conv2 = CausalConv1d(
                channels,
                channels,
                kernel_size=1,
                bias=bias,
            )
        else:
            self.pointwise_conv1 = nn.Conv1d(
                channels,
                2 * channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
            self.depthwise_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=channels,
                bias=bias,
            )
            self.norm = nn.BatchNorm1d(channels)
            self.pointwise_conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
