import torch.nn as nn
from stenunet.network_architecture.medicalnet_encoder_hf import MedicalNetResNet200Encoder

class WrappedMedicalNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MedicalNetResNet200Encoder(in_channels=1)

        # These must match what UNetDecoder expects
        self.output_channels = [256, 512, 1024, 2048]
        self.strides = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.conv_op = nn.Conv3d
        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0.0, 'inplace': True}
        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.kernel_sizes = [[3, 3, 3]] * 4
        self.conv_bias = False

    def forward(self, x):
        return self.encoder(x)  # returns list of feature maps
