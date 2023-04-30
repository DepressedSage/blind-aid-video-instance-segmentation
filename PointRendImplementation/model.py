import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.optim as optim

class PointRend(nn.Module):
    def __init__(self, in_channels, out_channels, num_points):
        super(PointRend, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = num_points

        # ResNet50 backbone
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.point_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_points, kernel_size=1, stride=1),
        )
        self.conv_out = nn.Conv2d(in_channels + num_points, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.backbone(x)

        points = F.adaptive_max_pool2d(out, (self.num_points, 1)).view(-1, 2048, self.num_points, 1)
        points = self.point_head(points).view(-1, self.num_points, 1, 1)
        out = torch.cat([out, points.expand(-1, -1, -1, out.size(3))], dim=1)

        out = self.conv_out(out)
        out = F.relu(out, inplace=True)

        return out

