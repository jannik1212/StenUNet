import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from collections import OrderedDict
from nnunetv2.dynamic_network_architectures.building_blocks.medicalnet_resnet import resnet200

class MedicalNetResNet200Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Download pretrained weights
        weights_path = hf_hub_download(
            repo_id="TencentMedicalNet/MedicalNet-Resnet200",
            filename="resnet_200.pth"
        )

        # Load the ResNet-200 architecture
        self.encoder = resnet200(in_channels=in_channels)

        # Load weights manually
        state_dict = torch.load(weights_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)

        # Remove classifier (we want to use this as an encoder only)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        features = []
        x = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x); features.append(x)
        x = self.encoder.layer2(x); features.append(x)
        x = self.encoder.layer3(x); features.append(x)
        x = self.encoder.layer4(x); features.append(x)
        return features
