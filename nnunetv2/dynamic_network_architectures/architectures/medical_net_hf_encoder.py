import torch.nn as nn
from nnunetv2.dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.dynamic_network_architectures.building_blocks.medicalnet_encoder_wrapper import WrappedMedicalNetEncoder

class StenUNetPretrained(nn.Module):
    def __init__(self,
                 num_input_channels,             # 1
                 n_stages,                       # 2
                 features_per_stage,             # 3 ← not used for encoder
                 conv_op,                        # 4 ← not used for encoder
                 kernel_size,                    # 5 ← usually just 3
                 strides,                        # 6 ← nnU-Net planning suggestion
                 blocks_per_stage_encoder,       # 7 ← use as n_conv_per_stage
                 num_classes,                    # 8 ← passed as num_labels
                 blocks_per_stage_decoder,       # 9 ← ignored unless needed
                 **kwargs                     
                 ):
        super().__init__()

        print("✅ StenUNetPretrained __init__ (dynamic-ready)")
        print(f"→ n_stages: {n_stages}, num_classes: {num_classes}")

        # Load your pretrained encoder (with fixed output channels)
        self.encoder = WrappedMedicalNetEncoder()

        # Dynamically build decoder conv config
        if isinstance(blocks_per_stage_encoder, (list, tuple)):
            n_conv_per_stage = blocks_per_stage_encoder
        else:
            n_conv_per_stage = [blocks_per_stage_encoder] * (n_stages - 1)

        # Use dynamic decoder config + fixed encoder
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=False  # change if needed
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)