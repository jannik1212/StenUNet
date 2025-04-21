import torch.nn as nn
from nnunetv2.dynamic_network_architectures.building_blocks.medical_net_encoder_model import MedicalNetResNet200Encoder

class WrappedMedicalNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MedicalNetResNet200Encoder(in_channels=1)

        # ✅ Tell nnU-Net how many features each stage outputs
        self.output_channels = [64, 128, 256, 512]

        # ✅ Tell nnU-Net how the spatial size reduces (same length as output_channels - 1)
        self.strides = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

        # ✅ These attributes are needed by the UNetDecoder and planner
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0.0, 'inplace': True}
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.kernel_sizes = [[3, 3, 3]] * len(self.output_channels)
        self.conv_bias = False

    def forward(self, x):
        return self.encoder(x)  # must return list of feature maps (skip connections)