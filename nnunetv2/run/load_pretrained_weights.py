import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!

    network can be either a plain model or DDP. We need to account for that in the parameter names
    """
    # 1) Load full checkpoint (allow numpy scalars etc.)
    saved = torch.load(fname, map_location="cpu", weights_only=False)
    pretrained = saved['network_weights']
    is_ddp = isinstance(network, DDP)

    # 2) Grab your model's own state dict
    model_dict = network.state_dict()

    # 3) Build a filtered dict of only those weights whose shape matches
    to_load = {}
    for pre_key, weight in pretrained.items():
        key = f"module.{pre_key}" if is_ddp else pre_key
        if key in model_dict:
            if model_dict[key].shape == weight.shape:
                to_load[key] = weight
            elif verbose:
                print(f"Skipping {key}: pretrained shape {tuple(weight.shape)}, model shape {tuple(model_dict[key].shape)}")
        elif verbose:
            print(f"Pretrained key not found in model: {key}")

    # 4) Update and load
    model_dict.update(to_load)
    print(f"################### Loaded {len(to_load)}/{len(model_dict)} matching parameters from {fname} ###################")
    if verbose:
        print("Parameters actually loaded:")
        for k in to_load:
            print("  ", k)

    network.load_state_dict(model_dict)
