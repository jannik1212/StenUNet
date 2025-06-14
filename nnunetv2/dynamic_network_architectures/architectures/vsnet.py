# dynamic_network_architectures/architectures/vsnet.py
"""
VSNet integration for nnU-Net pipeline.
Matches nnUNet dynamic-architecture constructor signature, but
adds tunable Transformer-style knobs (depth, heads, drop rates, etc).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type
from torch.nn.modules.conv import _ConvNd

from nnunetv2.dynamic_network_architectures.building_blocks.attention_blocks_vsnet import CSA, SSA
from nnunetv2.dynamic_network_architectures.building_blocks.conv_blocks_vsnet import DepTran, Gate
from nnunetv2.dynamic_network_architectures.building_blocks.swin_blocks_vsnet import SwinLayer, PatchMerging
from nnunetv2.dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp
)
from nnunetv2.dynamic_network_architectures.building_blocks.unetr_blocks import ConvStackND


class VSNet(nn.Module):
    """
    Wrapper so Trainer sees the expected API.
    Exposes extra VSNet transformer knobs for easy tuning.
    """
    def __init__(
        self,
        input_channels:  int,
        n_stages:         int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op:          Type[_ConvNd],
        kernel_sizes:     Union[int, List[int], Tuple[int, ...]],
        strides:          Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes:      int,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias:        bool = False,
        norm_op:          Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs:   dict = None,
        dropout_op:       Type[nn.Module] = None,
        dropout_op_kwargs:dict = None,
        nonlin:           Type[nn.Module] = nn.GELU,
        nonlin_kwargs:    dict = None,
        deep_supervision: bool = False,
        # newly exposed VSNet knobs:
        swin_depth:       int   = 2,      # how many Swin blocks (default = last encoder convs)
        num_heads:        int   = 8,      # number of attention heads
        attn_drop_rate:   float = 0.1,       # dropout inside attention
        dropout_path_rate:float = 0.1,       # stochastic depth in Swin
        use_checkpoint:   bool  = False      # gradient checkpointing in Swin
    ):
        super().__init__()
        # instantiate the core network, forwarding all the extra knobs:
        self.net = VSNetCore(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            swin_depth=swin_depth,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint
        )

        # pipeline-compatible handles
        self.encoders   = self.net.encoders
        self.pooling    = self.net.pooling
        self.bottleneck = self.net.swin
        self.upconvs    = self.net.upconvs
        self.decoders   = self.net.decoders
        heads = [self.net.seg_head]
        if deep_supervision:
            heads += [self.net.ds2, self.net.ds3]
        self.seg_layers = nn.ModuleList(heads)

        self.decoder = nn.Module()
        self.decoder.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
      return self.net.compute_conv_feature_map_size(input_size)


class VSNetCore(nn.Module):
    """
    VSNet main network, wired to nnU-Net dynamic pipeline signature,
    now with tunable Swin/attention hyper-parameters.
    """
    def __init__(
        self,
        input_channels:  int,
        n_stages:         int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op:          Type[_ConvNd],
        kernel_sizes:     Union[int, List[int], Tuple[int, ...]],
        strides:          Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes:      int,
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_bias:        bool = False,
        norm_op:          Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs:   dict = None,
        dropout_op:       Type[nn.Module] = None,
        dropout_op_kwargs:dict = None,
        nonlin:           Type[nn.Module] = nn.GELU,
        nonlin_kwargs:    dict = None,
        deep_supervision: bool = False,
        # forwarded VSNet knobs:
        swin_depth:       int = 2,
        num_heads:        int = 3,
        attn_drop_rate:   float = 0.1,
        dropout_path_rate:float = 0.1,
        use_checkpoint:   bool = False
    ):
        super().__init__()
        assert convert_conv_op_to_dim(conv_op) == 3, "VSNetCore supports only 3D"

        # normalize all the list-args to length 4
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        assert len(features_per_stage) == 4, "need exactly 4 stages for VSNet"
        def _to_list(x, length): return x if isinstance(x, (list,tuple)) else [x]*length

        feats      = list(features_per_stage)
        pools      = _to_list(strides, 4)
        kernels    = _to_list(kernel_sizes, 4)
        enc_blocks = _to_list(n_conv_per_stage, 4)
        if isinstance(n_conv_per_stage_decoder, (list, tuple)):
            dec_blocks = list(n_conv_per_stage_decoder) + [n_conv_per_stage_decoder[-1]]
        else:
            dec_blocks = [n_conv_per_stage_decoder] * 4
        
        dec_blocks = dec_blocks[:4]

        # capture for compute_conv_feature_map_size
        self._features    = feats
        self._strides     = pools
        self._num_classes = num_classes

        # --- Encoder ---
        self.encoders = nn.ModuleList([
            ConvStackND(
                in_ch         = input_channels if i==0 else feats[i-1],
                out_ch        = feats[i],
                num_convs     = enc_blocks[i],
                conv_op       = conv_op,
                conv_bias     = conv_bias,
                norm_op       = norm_op,
                norm_kwargs   = norm_op_kwargs,
                dropout_op    = dropout_op,
                dropout_op_kwargs= dropout_op_kwargs,
                nonlin        = nonlin,
                nonlin_kwargs = nonlin_kwargs
            )
            for i in range(4)
        ])
        self.pooling = nn.ModuleList([
            nn.Identity(),
            nn.MaxPool3d(pools[1]),
            nn.MaxPool3d(pools[2]),
            nn.MaxPool3d(pools[3]),
        ])
        self.gates = nn.ModuleList([
            Gate(feats[i-1], feats[i], feats[i-1]) for i in range(1,4)
        ])

        pm = PatchMerging(
            dim=feats[-1],
            norm_layer=norm_op,
            spatial_dims=convert_conv_op_to_dim(conv_op)
        )

        # --- Bottleneck: Swin Transformer + CSA + SSA ---
        self.swin = SwinLayer(
            dim               = feats[-1],
            depth             = swin_depth or enc_blocks[-1],
            num_heads         = num_heads or (feats[-1]//feats[0]),
            window_size       = (7,7,7),
            attn_drop         = attn_drop_rate,
            drop_path         = dropout_path_rate,
            downsample        = pm,
            use_checkpoint    = use_checkpoint
        )

        merged_channels = feats[-1]*2

        self.CSA = CSA(
            in_chans     = merged_channels,
            dropout_rate = attn_drop_rate
        )
        self.SSA = SSA(
            hidden_size  = merged_channels,
            num_heads    = num_heads or (feats[-1]//feats[0]),
            dropout_rate = attn_drop_rate,
            qkv_bias     = True
        )

        # --- Depth‐transfer between scales ---
        dt_channels = [merged_channels] + feats[::-1][1:]
        self.dt = nn.ModuleList([DepTran(c, c) for c in dt_channels])

        # --- Decoder (transpose-conv + conv-stack) + five heads ---
        transp = get_matching_convtransp(conv_op)
        # after dt_channels = [512,256,128,64], and feats = [32,64,128,256]
        up_out_ch = [feats[2], feats[1], feats[0], feats[0]]  # [128, 64, 32, 32]
        # kernels and pools are already lists of length 4
        self.upconvs = nn.ModuleList([
            transp(
                in_channels   = dt_channels[i],
                out_channels  = up_out_ch[i],
                kernel_size   = kernels[3-i],
                stride        = pools[3-i],
                bias          = conv_bias
            )
            for i in range(4)
        ])


        # build decoder conv‐stacks with correct in_ch = skip_ch + upconv_ch
        self.decoders = nn.ModuleList()
        for i in range(4):
            # how many channels came out of upconv i?
            up_ch = [up_out_ch[0], up_out_ch[1], up_out_ch[2], up_out_ch[3]][i]
            # how many channels are in the corresponding skip?
            skip_ch = [feats[3], feats[2], feats[1], feats[0]][i]
            in_ch   = skip_ch + up_ch
            out_ch  = (feats[2-i] if i < 3 else feats[0])
            self.decoders.append(
                ConvStackND(
                    in_ch         = in_ch,
                    out_ch        = out_ch,
                    num_convs     = dec_blocks[3-i],
                    conv_op       = conv_op,
                    conv_bias     = conv_bias,
                    norm_op       = norm_op,
                    norm_op_kwargs= norm_op_kwargs,
                    dropout_op    = dropout_op,
                    dropout_op_kwargs= dropout_op_kwargs,
                    nonlin        = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                )
            )

        # → raw logits (no Softmax baked in)
        # full_res = ups[3] has feats[0] channels
        self.seg_head = nn.Conv3d(
            in_channels=feats[0],
            out_channels=num_classes,
            kernel_size=1,
            bias=True
        )
        if deep_supervision:
            # mid_res = ups[2] also has feats[0] channels
            self.ds2 = nn.Conv3d(
                in_channels=feats[0],
                out_channels=num_classes,
                kernel_size=1,
                bias=True
            )
            # small_res = ups[1] has feats[1] channels
            self.ds3 = nn.Conv3d(
                in_channels=feats[1],
                out_channels=num_classes,
                kernel_size=1,
                bias=True
            )


        # dummy decoder attr for nnU-Net compatibility
        self.decoder = nn.Module()
        self.decoder.deep_supervision = deep_supervision
    
    def forward(self, x: torch.Tensor):
      # input
      print(f"[NAN TRACE] input        = {tuple(x.shape)}, "
            f"min/max/mean = {x.min().item():.3e}/{x.max().item():.3e}/{x.mean().item():.3e}")

      # ENCODER + gating
      x1 = self.encoders[0](x)
      print(f"[NAN TRACE] enc1 out     = {tuple(x1.shape)}, "
            f"min/max/mean = {x1.min().item():.3e}/{x1.max().item():.3e}/{x1.mean().item():.3e}")

      x2 = self.encoders[1](x1)
      x2p = self.pooling[1](x2)
      x1g = self.gates[0](x1, x2p)
      print(f"[NAN TRACE] gate1 out    = {tuple(x1g.shape)}, "
            f"min/max/mean = {x1g.min().item():.3e}/{x1g.max().item():.3e}/{x1g.mean().item():.3e}")

      x3 = self.encoders[2](x2p)
      x3p = self.pooling[2](x3)
      x2g = self.gates[1](x2p, x3p)
      print(f"[NAN TRACE] gate2 out    = {tuple(x2g.shape)}, "
            f"min/max/mean = {x2g.min().item():.3e}/{x2g.max().item():.3e}/{x2g.mean().item():.3e}")

      x4 = self.encoders[3](x3p)
      x4p = self.pooling[3](x4)
      x3g = self.gates[2](x3p, x4p)
      print(f"[NAN TRACE] gate3 out    = {tuple(x3g.shape)}, "
            f"min/max/mean = {x3g.min().item():.3e}/{x3g.max().item():.3e}/{x3g.mean().item():.3e}")

      # BOTTLENECK
      print(f"[NAN TRACE] bottleneck in = {tuple(x4p.shape)}, "
            f"min/max/mean = {x4p.min().item():.3e}/{x4p.max().item():.3e}/{x4p.mean().item():.3e}")

      x5 = self.swin(x4p)
      print(f"[NAN TRACE] after Swin    = {tuple(x5.shape)}, "
            f"min/max/mean = {x5.min().item():.3e}/{x5.max().item():.3e}/{x5.mean().item():.3e}")

      x5 = self.CSA(x5)
      print(f"[NAN TRACE] after CSA     = {tuple(x5.shape)}, "
            f"min/max/mean = {x5.min().item():.3e}/{x5.max().item():.3e}/{x5.mean().item():.3e}")

      x5 = self.SSA(x5)
      print(f"[NAN TRACE] after SSA     = {tuple(x5.shape)}, "
            f"min/max/mean = {x5.min().item():.3e}/{x5.max().item():.3e}/{x5.mean().item():.3e}")

      # DECODER
      skips = [x4, x3g, x2g, x1g]
      ups   = []
      out   = x5
      for i in range(4):
          out = self.dt[i](out)
          print(f"[NAN TRACE] dt[{i}] out    = {tuple(out.shape)}, "
                f"min/max/mean = {out.min().item():.3e}/{out.max().item():.3e}/{out.mean().item():.3e}")

          out = self.upconvs[i](out)
          print(f"[NAN TRACE] upconv[{i}] out= {tuple(out.shape)}, "
                f"min/max/mean = {out.min().item():.3e}/{out.max().item():.3e}/{out.mean().item():.3e}")

          sk  = skips[i]
          if out.shape[2:] != sk.shape[2:]:
              out = F.interpolate(
                  out,
                  size=sk.shape[2:],
                  mode='trilinear',
                  align_corners=False
              )
              print(f"[NAN TRACE] interp[{i}]   = {tuple(out.shape)}, "
                    f"min/max/mean = {out.min().item():.3e}/{out.max().item():.3e}/{out.mean().item():.3e}")

          out = torch.cat([sk, out], dim=1)
          print(f"[NAN TRACE] concat[{i}]   = {tuple(out.shape)}, "
                f"min/max/mean = {out.min().item():.3e}/{out.max().item():.3e}/{out.mean().item():.3e}")

          out = self.decoders[i](out)
          print(f"[NAN TRACE] decoder[{i}] out= {tuple(out.shape)}, "
                f"min/max/mean = {out.min().item():.3e}/{out.max().item():.3e}/{out.mean().item():.3e}")

          ups.append(out)

      # unpack resolutions
      low_res, small_res, mid_res, full_res = ups
      print(f"[NAN TRACE] full_res      = {tuple(full_res.shape)}")

      # heads
      main = self.seg_head(full_res)
      print(f"[NAN TRACE] seg_head out  = {tuple(main.shape)}, "
            f"min/max/mean = {main.min().item():.3e}/{main.max().item():.3e}/{main.mean().item():.3e}")

      if self.decoder.deep_supervision:
          ds2 = self.ds2(mid_res)
          print(f"[NAN TRACE] ds2 out       = {tuple(ds2.shape)}, "
                f"min/max/mean = {ds2.min().item():.3e}/{ds2.max().item():.3e}/{ds2.mean().item():.3e}")

          ds3 = self.ds3(small_res)
          print(f"[NAN TRACE] ds3 out       = {tuple(ds3.shape)}, "
                f"min/max/mean = {ds3.min().item():.3e}/{ds3.max().item():.3e}/{ds3.mean().item():.3e}")

          return [main, ds2, ds3]

      return main


    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> int:
        # identical to before, for VRAM planning
        sizes = list(input_size)
        total = 0
        # encoder
        for stage in range(4):
            c = self._features[stage]
            total += c * int(torch.prod(torch.tensor(sizes)))
            if stage < 3:
                for d in range(len(sizes)):
                    sizes[d] //= self._strides[stage + 1]
        # bottleneck
        c = self._features[-1]
        total += c * int(torch.prod(torch.tensor(sizes)))
        # decoder & heads
        dec_feats = [self._features[2], self._features[1], self._features[0], self._features[0]]
        for i in range(4):
            for d in range(len(sizes)):
                sizes[d] *= self._strides[-(i+1)]
            total += dec_feats[i] * int(torch.prod(torch.tensor(sizes)))
            total += self._num_classes * int(torch.prod(torch.tensor(sizes)))
        return int(total)
