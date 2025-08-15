import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention mechanism with spatial feature recalibration"""
    def __init__(self, in_channels, reduction=4):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pool = nn.AvgPool2d(kernel_size=reduction, stride=reduction)

    def forward(self, x):
        # Spatial attention computation
        x_pooled = self.pool(x)
        proj_query = self.query_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))
        proj_key = self.key_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))
        proj_value = self.value_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention)
        out = out.view(x_pooled.size())
        
        # Upsample attention map
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
        return self.gamma * out + x