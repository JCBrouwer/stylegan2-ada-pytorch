from timeit import timeit as time

from tqdm import tqdm
from training.networks import Generator

import torch
import tvm
from tvm import relay

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

batch_size = 1
input_shape = (batch_size, 512)
output_shape = (batch_size, 3, 1024, 1024)

device = "cuda"


G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3).float().eval().to(device)
G = torch.jit.trace(G, torch.randn(input_shape, device=device)).eval()
for _ in range(5):
    G(torch.randn(input_shape, device=device))  # warm up cudnn autotuner


def randn(inputs, input_types):
    return tvm.relay.expr.const(
        torch.randn(
            size=tuple(int(i.data.asnumpy()) if isinstance(i, tvm.relay.Constant) else int(i) for i in inputs[0])
        ).numpy()
    )


mod, params = relay.frontend.from_pytorch(G, [("input", input_shape)], {"aten::randn": randn})

print("Quantizing...")
with tvm.transform.PassContext(opt_level=3):
    mod = relay.transform.FoldConstant()(mod)
with relay.quantize.qconfig(calibrate_mode="global_scale"):
    mod = relay.quantize.quantize(mod, params)
qG = relay.create_executor("vm", mod, tvm.device(device), device).evaluate()


print("PyTorch")
print(time(lambda: G(torch.randn(size=input_shape, device=device)), number=100) * 10, "ms")

print("Quantized")
print(time(lambda: qG(torch.randn(size=input_shape)), number=100) * 10, "ms")
