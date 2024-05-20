import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F

class SmallUNet(nn.Module):
    def __init__(self):
        super(SmallUNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  #scala di grigi

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        
        # Decoder
        self.upconv3 = nn.Conv2d(384, 128, 3, padding=1)
        self.upconv2 = nn.Conv2d(192, 64, 3, padding=1)
        self.upconv1 = nn.Conv2d(96, 32, 3, padding=1)
        self.final_conv = nn.Conv2d(32, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        conv1 = F.relu(self.conv1(x))
        x = self.maxpool(conv1)
        
        conv2 = F.relu(self.conv2(x))
        x = self.maxpool(conv2)
        
        conv3 = F.relu(self.conv3(x))
        x = self.maxpool(conv3)
        
        x = F.relu(self.conv4(x))
        
        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = F.relu(self.upconv3(x))
        
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = F.relu(self.upconv2(x))
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = F.relu(self.upconv1(x))
        
        out = torch.sigmoid(self.final_conv(x))
        
        return out
    