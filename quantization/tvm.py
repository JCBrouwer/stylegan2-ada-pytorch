from timeit import timeit as time

import dnnlib
import legacy
import torchvision as tv
from training.dataset import ImageFolderDataset

import torch
import tvm
from tvm import relay

batch_size = 1
data_shape = (batch_size, 512)
model_name = "StyleGAN"
device = "cuda"
tvmdev = tvm.device(device)

with dnnlib.util.open_url("afhqwild.pkl") as fp:
    net_dict = legacy.load_network_pkl(fp)
    generator = net_dict["G"].requires_grad_(False).to(device)


generator = torch.jit.trace(generator, torch.randn(data_shape, device=device)).eval()


def randn(inputs, input_types):
    return tvm.relay.expr.const(torch.randn(size=(1, *inputs[0][1:])).numpy())


print(generator.synthesis.b256.conv0)
print(generator.synthesis.b256.conv0.code)
print(generator.synthesis.b256.conv0.graph)

mod, params = relay.frontend.from_pytorch(generator, [("input", data_shape)], {"aten::randn": randn})
# torch.onnx.export(generator, torch.randn(data_shape, device=device), "generator.onnx", export_params=True, verbose=True)
# mod, params = relay.frontend.from_onnx(onnx.load("generator.onnx"), shape={"0": data_shape})

print(mod)

loader = torch.utils.data.DataLoader(
    ImageFolderDataset("/home/hans/datasets/naomo/1024/"), num_workers=8, batch_size=2,
)
with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
    mod = relay.quantize.quantize(mod, params, dataset=loader)
print(mod)
executor = relay.create_executor("vm", mod, tvmdev, device)

inputs = torch.randn(size=data_shape, device=device)
# compare the performance
print("PyTorch")
print(time(lambda: generator(inputs), number=100))
print("Quantized")
print(time(lambda: executor.evaluate()(inputs), number=100))

imgs = []
for _ in range(18):
    imgs.append(executor.evaluate()(torch.randn(size=data_shape)))
imgs = torch.cat(imgs)
tv.utils.save_image(tv.utils.make_grid(imgs, nrow=6), "quantized.jpg")

imgs = generator(torch.randn(size=(18, 512), device=device))
tv.utils.save_image(tv.utils.make_grid(imgs, nrow=6), "original.jpg")
