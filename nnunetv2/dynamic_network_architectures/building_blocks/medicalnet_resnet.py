import torch.nn as nn

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

def make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes)
        )
    layers = [block(inplanes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)

class MedicalNetResNet200(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, layers=[3, 24, 36, 3]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(BasicBlock, 64, 64, layers[0])
        self.layer2 = make_layer(BasicBlock, 64, 128, layers[1], stride=2)
        self.layer3 = make_layer(BasicBlock, 128, 256, layers[2], stride=2)
        self.layer4 = make_layer(BasicBlock, 256, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet200(in_channels=1, num_classes=2):
    return MedicalNetResNet200(in_channels=in_channels, num_classes=num_classes)
