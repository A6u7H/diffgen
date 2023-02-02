import torch
import torch.nn as nn

from layers import (
    DownBlock,
    UpBlock,
    DoubleConv,
    SelfAttention
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
        device: str = "cuda"
    ) -> None:
        super(UNet, self).__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = DownBlock(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = DownBlock(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = UpBlock(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = UpBlock(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = UpBlock(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
