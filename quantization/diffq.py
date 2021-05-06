import os
from copy import copy
from timeit import timeit as time

import dnnlib
import legacy
import torchvision as tv
from tqdm import tqdm
from training.dataset import ImageFolderDataset

import torch
import torch.quantization
from diffq import DiffQuantizer
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
device = "cuda:0"

with dnnlib.util.open_url(
    "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl"
) as fp:
    net_dict = legacy.load_network_pkl(fp)
    D = net_dict["D"].requires_grad_(False).to(device)
    G = net_dict["G"].requires_grad_(False).to(device)

loader = torch.utils.data.DataLoader(
    ImageFolderDataset("/home/hans/datasets/naomo/1024/"), num_workers=8, batch_size=2,
)

opt_G = torch.optim.Adam(G.parameters())
opt_D = torch.optim.Adam(D.parameters())

quantizer = DiffQuantizer(G)
quantizer.opt = torch.optim.Adam([{"params": []}])
quantizer.setup_optimizer(quantizer.opt)

penalty = 1e-2
for i, (real_imgs, labels) in enumerate(tqdm(loader)):
    G.requires_grad_(True)
    D.requires_grad_(False)
    opt_G.zero_grad()
    quantizer.opt.zero_grad()

    z = torch.randn(size=(len(real_imgs), 512), device=device)
    ws = G.mapping(z, c=None)
    imgs = G.synthesis(ws)
    logits = D(imgs, c=None)
    loss_G = F.softplus(-logits).mean() + penalty * quantizer.model_size()

    loss_G.backward()
    opt_G.step()
    quantizer.opt.step()

    G.requires_grad_(False)
    D.requires_grad_(True)
    opt_D.zero_grad()
    z = torch.randn(size=(len(real_imgs), 512), device=device)
    ws = G.mapping(z, c=None)
    imgs = G.synthesis(ws)
    logits_fake = D(imgs, c=None)
    logits_real = D(real_imgs.detach().to(device), None)
    loss_D = F.softplus(logits_fake).mean() + F.softplus(-logits_real).mean()
    loss_D.backward()
    opt_D.step()

    if i == 10:
        break

quantizer.quantize()  # why does this error?!?
qG = copy(G)
print(qG)

with dnnlib.util.open_url(
    "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl"
) as fp:
    G = net_dict["G_ema"].requires_grad_(False).to(device)


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, " \t", "Size (KB):", size / 1e3)
    os.remove("temp.p")
    return size


# compare the sizes
f = print_size_of_model(G, "original")
q = print_size_of_model(qG, "quantized")
print("{0:.2f} times smaller".format(f / q))

inputs = torch.randn(size=(1, 512), device=device)

print("Floating point FP32")
print(time(lambda: G(inputs, c=None), number=100))
print("Quantized INT8")
print(time(lambda: qG(inputs, c=None), number=100))

# run the float model
out1 = G(inputs, c=None)
mag1 = torch.mean(abs(out1)).item()
print("mean absolute value of output tensor values in the FP32 model is {0:.5f} ".format(mag1))
out2 = qG(inputs, c=None)
mag2 = torch.mean(abs(out2)).item()
print("mean absolute value of output tensor values in the INT8 model is {0:.5f}".format(mag2))
mag3 = torch.mean(abs(out1 - out2)).item()
print(
    "mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent".format(
        mag3, mag3 / mag1 * 100
    )
)

imgs = qG(torch.randn(size=(18, 512), device=device), c=None)
grid = tv.utils.make_grid(imgs, nrow=6, normalize=True)
tv.utils.save_image(grid, "quantized.jpg")

imgs = G(torch.randn(size=(18, 512), device=device), c=None)
grid = tv.utils.make_grid(imgs, nrow=6, normalize=True)
tv.utils.save_image(grid, "original.jpg")
