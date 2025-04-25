import torch.nn as nn
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.dynamic_network_architectures.architectures.resnet import ResNet200_MedicalNet


class ResNet200UNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage,
                 conv_op,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage_decoder,
                 num_classes: int,
                 deep_supervision: bool = False):
        super().__init__()

        # Instantiate the encoder (ResNet200 with MedicalNet weights)
        self.encoder = ResNet200_MedicalNet(
            n_classes=num_classes,  # not used in encoder, but required by constructor
            n_input_channels=input_channels,
            input_dimension=len(kernel_sizes[0])
        ).encoder  # <-- extract only the encoder from the model

        self.encoder.return_skips = True  # ensure skip connection outputs

        # Attach UNet decoder
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
