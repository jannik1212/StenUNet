from nnunetv2.dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.dynamic_network_architectures.building_blocks.medicalnet_encoder_wrapper import WrappedMedicalNetEncoder
import torch.nn as nn

class StenUNetPretrained(nn.Module):
    def __init__(self, num_classes, n_conv_per_stage, deep_supervision):
        super().__init__()
        self.encoder = WrappedMedicalNetEncoder()
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
