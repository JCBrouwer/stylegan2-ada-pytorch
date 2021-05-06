import os
from copy import copy
from timeit import timeit as time

import dnnlib
import legacy
import torchvision as tv

import torch
import torch.quantization

torch.backends.cudnn.benchmark = True
device = "cuda"


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (KB):", size / 1e3)
    os.remove("temp.p")
    return size


with dnnlib.util.open_url(
    "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl"
) as fp:
    net_dict = legacy.load_network_pkl(fp)
    G = net_dict["G"].requires_grad_(False).to(device)

qG = copy(G)
qG.qconfig = torch.quantization.default_qconfig
print(qG.qconfig)
torch.quantization.prepare(qG, inplace=True)
with torch.no_grad():
    for _ in range(32 * 16):
        z = torch.randn(size=(1, 512), device=device)
        ws = G.mapping(z, c=None)
        imgs = G.synthesis(ws)
torch.quantization.convert(qG, inplace=True)

# compare the sizes
f = print_size_of_model(G, "original")
q = print_size_of_model(qG, "quantized")
print("{0:.2f} times smaller".format(f / q))

print("Floating point FP32")
print(time(lambda: G(torch.randn(size=(1, 512), device=device), c=None), number=100))
print("Quantized INT8")
print(time(lambda: qG(torch.randn(size=(1, 512), device=device), c=None), number=100))

imgs = G(torch.randn(size=(24, 512), device=device), c=None)
tv.utils.save_image(tv.utils.make_grid(imgs, nrow=6), "original.jpg")

imgs = qG(torch.randn(size=(24, 512), device=device), c=None)
tv.utils.save_image(tv.utils.make_grid(imgs, nrow=6), "quantized.jpg")
