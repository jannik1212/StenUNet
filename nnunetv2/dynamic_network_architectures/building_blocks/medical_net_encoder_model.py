import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import torchvision.models.video.resnet as resnet3d
from collections import OrderedDict

class MedicalNetResNet200Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        weights_path = hf_hub_download(
            repo_id="TencentMedicalNet/MedicalNet-Resnet200",
            filename="resnet_200.pth"
        )

        self.encoder = resnet3d._resnet(
            'resnet200',
            resnet3d.Bottleneck,
            [3, 8, 36, 3],
            pretrained=False,
            progress=True
        )

        if in_channels != 3:
            self.encoder.stem[0] = nn.Conv3d(
                in_channels,
                self.encoder.stem[0].out_channels,
                kernel_size=self.encoder.stem[0].kernel_size,
                stride=self.encoder.stem[0].stride,
                padding=self.encoder.stem[0].padding,
                bias=False
            )

        state_dict = torch.load(weights_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)

        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()

    def forward(self, x):
        features = []
        x = self.encoder.stem(x)
        x = self.encoder.layer1(x); features.append(x)
        x = self.encoder.layer2(x); features.append(x)
        x = self.encoder.layer3(x); features.append(x)
        x = self.encoder.layer4(x); features.append(x)
        return features
