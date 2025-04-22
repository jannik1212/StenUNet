import torch
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

        # Step 1: Load encoder
        self.encoder = WrappedMedicalNetEncoder()

        # Force n_stages to match encoder output
        n_stages = len(self.encoder.output_channels)
        print("✅ StenUNetPretrained __init__ (dynamic-ready)")
        print(f"→ forced n_stages: {n_stages}, num_classes: {num_classes}")

        # Validate number of decoder stages = len(strides) = len(output_channels) - 1
        assert len(self.encoder.strides) == len(self.encoder.output_channels) - 1, \
            f"Mismatch: {len(self.encoder.strides)} strides for {len(self.encoder.output_channels)} outputs"

        num_decoder_stages = len(self.encoder.strides)
        print(f"→ decoder stages: {num_decoder_stages}")

        # Generate correct n_conv_per_stage
        if isinstance(blocks_per_stage_encoder, int):
            n_conv_per_stage = [blocks_per_stage_encoder] * num_decoder_stages
        elif isinstance(blocks_per_stage_encoder, (list, tuple)):
            n_conv_per_stage = list(blocks_per_stage_encoder[:num_decoder_stages])
            while len(n_conv_per_stage) < num_decoder_stages:
                n_conv_per_stage.append(n_conv_per_stage[-1])
        else:
            raise ValueError("blocks_per_stage_encoder must be int or list/tuple")

        print(f"→ n_conv_per_stage = {n_conv_per_stage}")

        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=True  # You can toggle this if needed
        )

    def forward(self, x):
        skips = self.encoder(x)
        out = self.decoder(skips)
        return out if isinstance(out, (tuple, list)) else [out]

    
    def compute_conv_feature_map_size(self, input_size):
        """
        Overrides default nnU-Net static VRAM estimation.
        Ensures compatibility without modifying base files.
        """
        # Manually compute skip sizes
        skip_sizes = []
        for stride in self.encoder.strides:
            input_size = [i // s for i, s in zip(input_size, stride)]
            skip_sizes.append(input_size)

        # Make sure skip sizes match decoder stages
        if len(skip_sizes) > len(self.decoder.stages):
            skip_sizes = skip_sizes[:len(self.decoder.stages)]

        total = 0
        for s in range(len(self.decoder.stages)):
            # conv blocks
            total += self.decoder.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            total += torch.prod(torch.tensor([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]]), dtype=torch.int64).item()
            # segmentation layer
            if self.decoder.deep_supervision or (s == (len(self.decoder.stages) - 1)):
                total += torch.prod(torch.tensor([self.decoder.num_classes, *skip_sizes[-(s+1)]]), dtype=torch.int64).item()

        return total

