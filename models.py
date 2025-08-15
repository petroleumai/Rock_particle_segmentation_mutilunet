import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """Basic UNet block with double convolution and batch normalization"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    """Downsampling block with max pooling"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = UNetBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.pool(x1)
        return x1, x2

class UpBlock(nn.Module):
    """Upsampling block with transpose convolution"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = UNetBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)

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
        x_pooled = self.pool(x)
        proj_query = self.query_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))
        proj_key = self.key_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))
        proj_value = self.value_conv(x_pooled).view(x_pooled.size(0), -1, x_pooled.size(2) * x_pooled.size(3))
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = torch.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention)
        out = out.view(x_pooled.size())
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
        return self.gamma * out + x

class PMFSNetWithUNet(nn.Module):
    """Complete UNet-based segmentation model with multi-scale features and attention"""
    def __init__(self, in_channels=3, num_views=5, num_classes=1):
        super(PMFSNetWithUNet, self).__init__()
        self.num_views = num_views

        # Multi-scale feature extraction
        self.scale1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.scale2 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.scale3 = nn.Conv2d(in_channels, 64, kernel_size=7, padding=3)

        # UNet encoder
        self.down1 = DownBlock(64 * 3 * num_views, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        
        self.bottleneck = UNetBlock(512, 1024)
        
        # UNet decoder
        self.up3 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up1 = UpBlock(256, 128)

        # Attention mechanism
        self.attention = SelfAttention(128)

        # Output head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        view_features = []
        for i in range(self.num_views):
            view = x[:, i, :, :, :]
            x1 = F.relu(self.scale1(view))
            x2 = F.relu(self.scale2(view))
            x3 = F.relu(self.scale3(view))
            weight = 2 if i > 0 else 1
            view_feature = torch.cat([x1, x2, x3], dim=1) * weight
            view_features.append(view_feature)

        fused = torch.cat(view_features, dim=1)
        
        # UNet encoding
        x1, x = self.down1(fused)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # Attention
        x = self.attention(x)

        # Output
        segmentation_result = self.segmentation_head(x)
        upsampled_output = F.interpolate(segmentation_result, size=(256, 256), mode='bilinear', align_corners=False)
        return upsampled_output