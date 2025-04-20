import torch.nn as nn
from nnunetv2.dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.dynamic_network_architectures.building_blocks.medicalnet_encoder_wrapper import WrappedMedicalNetEncoder

class StenUNetPretrained(nn.Module):
    def __init__(self,
                 num_input_channels,                     
                 n_stages,                               
                 features_per_stage,                     
                 conv_op,                                
                 kernel_sizes,                           
                 strides,                                
                 num_classes,                            
                 deep_supervision,                       
                 precomputed_pool_op_kernel_sizes,       
                 conv_kernel_sizes,                     
                 **kwargs                                
                 ):
        super().__init__()

        # Pretrained encoder (output_channels must match decoder expectations)
        self.encoder = WrappedMedicalNetEncoder()

        # Dynamically compute n_conv_per_stage based on stages
        # nnU-Net uses either fixed or variable convs per stage
        if isinstance(kernel_sizes[0], list):  # per-stage kernel lists
            n_conv_per_stage = [len(k) for k in kernel_sizes]
        else:
            n_conv_per_stage = [len(kernel_sizes)] * (n_stages - 1)

        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
