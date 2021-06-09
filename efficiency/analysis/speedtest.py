import gc
from time import time

import numpy as np
import torch

import dnnlib
from training.networks import Generator


def get_model_size(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    module(*inputs)
    for hook in hooks:
        hook.remove()

    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    param_count = 0
    param_size = 0
    intermediate_count = 0
    intermediate_size = 0
    for e in entries:
        num_params = sum(t.numel() for t in e.unique_params)
        num_buffer = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [int(_) for _ in e.outputs[0].shape]
        output_dtypes = [int(str(t.dtype).split(".")[-1].replace("float", "")) for t in e.outputs]
        param_count += num_params + num_buffer
        param_size += (num_params + num_buffer) * output_dtypes[0]
        intermediate_count += np.prod(output_shapes)
        intermediate_size += np.prod(output_shapes) * output_dtypes[0]

    return param_count, param_size, intermediate_count, intermediate_size


def measure_inference_latency(G_map, G_synth, input_size=(16, 512), num_samples=100, num_warmups=10):
    x = torch.randn(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            ws = G_map(x, None)
            _ = G_synth(ws, force_fp32=device == "cpu")
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time()
        for _ in range(num_samples):
            ws = G_map(x, None)
            _ = G_synth(ws, force_fp32=device == "cpu")
            torch.cuda.synchronize()
        end_time = time()
    elapsed_time = end_time - start_time
    elapsed_time_avg = elapsed_time / num_samples

    return elapsed_time_avg


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False

    w_dim = 512
    channel_base = 32768
    channel_max = 512

    sparsities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(
        "sparsity".ljust(12),
        "w_dim".ljust(9),
        "channel_base".ljust(16),
        "channel_max".ljust(15),
        "param_count".ljust(15),
        "param_size".ljust(14),
        "intermediate_count".ljust(22),
        "intermediate_size".ljust(21),
        "time_per_batch".ljust(18),
    )
    for sparsity in sparsities:
        G = (
            Generator(
                z_dim=512,
                c_dim=0,
                w_dim=int(w_dim * (1 - sparsity)),
                img_resolution=256,
                img_channels=3,
                synthesis_kwargs={
                    "channel_base": int(channel_base * (1 - sparsity)),
                    "channel_max": int(channel_max * (1 - sparsity)),
                    "num_fp16_res": 0,
                },
            )
            .eval()
            .to(device)
        )
        param_count, param_size, intermediate_count, intermediate_size = get_model_size(
            G, (torch.randn((16, 512), device=device), None)
        )

        time_per_batch = measure_inference_latency(G.mapping, G.synthesis)

        del G
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"{sparsity}".ljust(12),
            f"{int(w_dim * (1 - sparsity))}".ljust(9),
            f"{int(channel_base * (1 - sparsity))}".ljust(16),
            f"{int(channel_max * (1 - sparsity))}".ljust(15),
            f"{param_count}".ljust(15),
            f"{param_size/8/1024/1024:.2f} MB".ljust(14),
            f"{intermediate_count}".ljust(22),
            f"{intermediate_size/8/1024/1024:.2f} MB".ljust(21),
            f"{time_per_batch:.3f}".ljust(18),
        )