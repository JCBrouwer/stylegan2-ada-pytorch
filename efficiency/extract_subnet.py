import sys

import dnnlib
import legacy
import torch
from training.networks import Generator

CLOSE_TO_ZERO = 1e-3


class MappingSubnet:
    pass


class SynthesisSubnet:
    pass


@torch.no_grad()
def extract_subnet(network_pkl, z_dim=512, c_dim=0, w_dim=512, img_resolution=256, img_channels=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = Generator(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_pkl(f)["G"].to(device)

    weights, masks, sizes, new_shapes = [], [], [], []
    for name, mod in list(G.mapping.named_modules()) + list(G.synthesis.named_modules()):
        if "fc" in name or "affine" in name:
            w = mod.weight
            m = mod.mask

            close_to_zero = abs(m) <= CLOSE_TO_ZERO

            sparsity = close_to_zero.sum() / w.shape[0]
            out_size = int(w.shape[0] - close_to_zero.sum())

            weights.append(w)
            masks.append(close_to_zero)
            sizes.append(out_size)
            new_shapes.append((out_size, sizes[-2] if len(sizes) > 1 else 512))

            print(
                f"{name}".ljust(20),
                f"{list(reversed(w.shape))}".ljust(16),
                "->",
                f"{[sizes[-2] if len(sizes) > 1 else 512, out_size]}".ljust(20),
                f"sparsity: {100 * sparsity:.2f}".ljust(20),
            )


if __name__ == "__main__":
    extract_subnet(sys.argv[1])
