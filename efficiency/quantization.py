import copy
from abc import abstractmethod

import torch
from training.networks import SynthesisBlock


class Quantization:
    @abstractmethod
    def forward_pre_hook(self, module, input):
        pass


class QuantizeLinear(torch.autograd.Function):
    """
    from https://github.com/VITA-Group/GAN-Slimming/blob/master/models/models.py
    """

    @staticmethod
    def forward(ctx, input, signed, nbits, max_val):
        """
        In the forward pass we apply the quantizer
        """
        assert max_val > 0
        int_max = 2 ** (nbits - 1) - 1 if signed else 2 ** nbits
        scale = max_val / int_max
        return input.div(scale).round_().mul_(scale)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output, None, None, None


class Linear(Quantization):
    def __init__(self, parent, input_signed=False, quantize_mapping=False, nbits=8, input_max=4):
        self.parent = parent
        self.input_signed = input_signed
        self.nbits = nbits
        self.input_max = input_max

        print("\nQuantizing layers")
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if (quantize_mapping and "fc" in name) or "affine" in name or "conv" in name:
                mod.register_forward_pre_hook(self.forward_pre_hook)
                print(name, mod.weight.shape)
        print()

    def forward_pre_hook(self, module, input):
        module.weight.data = QuantizeLinear.apply(module.weight, True, self.nbits, module.weight.abs().max().item())
        module.bias.data = QuantizeLinear.apply(module.bias, True, self.nbits, module.bias.abs().max().item())

        output = []
        for i in range(len(input)):
            max_val = self.input_max
            min_val = -max_val if self.input_signed else 0.0
            output.append(
                QuantizeLinear.apply(input[i].clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)
            )
        return tuple(output)


class QuantizeQGAN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, beta):
        return input.sub(beta).div_(alpha).round_()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class QGAN(Quantization):
    def __init__(self, parent, input_signed=True, quantize_mapping=True, nbits=8, input_max=4):
        self.parent = parent
        self.input_signed = input_signed
        self.nbits = nbits
        self.input_max = input_max

        print("\nQuantizing layers")
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if (quantize_mapping and "fc" in name) or "affine" in name or "conv" in name:
                self.prepare_qgan(mod, "weight")
                self.prepare_qgan(mod, "bias")
                mod.register_forward_pre_hook(self.forward_pre_hook)
                print(name, mod.weight.shape)
        print()

    def prepare_qgan(self, module, param_name):
        # keep the unquantized weights as a nn.Parameter for backprop
        setattr(module, f"unquant_{param_name}", copy.deepcopy(getattr(module, param_name)))
        # delete the original weight nn.Parameter
        delattr(module, param_name)
        # keep the quantized version in the original attribute as a tensor (this is used for forward pass)
        setattr(module, param_name, torch.tensor(getattr(module, f"unquant_{param_name}").data))

        # create scaling parameters to be optimized with EM
        setattr(module, f"alpha_{param_name}", torch.ones([], device=self.parent.device))
        setattr(module, f"beta_{param_name}", torch.zeros([], device=self.parent.device))

    def forward_pre_hook(self, module, input):
        module.weight.data = self.quantize_parameters(module, "weight")
        module.bias.data = self.quantize_parameters(module, "bias")
        # output = []
        # for i in range(len(input)):
        #     max_val = self.input_max
        #     min_val = -max_val if self.input_signed else 0.0
        #     output.append(
        #         QuantizeLinear.apply(input[i].clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)
        #     )
        # return tuple(output)

    def quantize_parameters(self, module, param_name):
        # maximization
        w = getattr(module, f"unquant_{param_name}")
        z = getattr(module, param_name)
        a = (torch.mean(w * z) - torch.mean(w) * torch.mean(z)) / (torch.mean(z * z) - torch.mean(z).square())
        b = torch.mean(w) - a * torch.mean(z)

        # update modules stored values
        getattr(module, f"alpha_{param_name}").data = a
        getattr(module, f"beta_{param_name}").data = b

        # print(module, param_name)
        # print(w)
        # print(z)
        # print(getattr(module, f"alpha_{param_name}").item(), getattr(module, f"beta_{param_name}").item())

        return QuantizeQGAN.apply(w, a, b)  # expectation


# automatic torch.quantization stuff doesn't seem to work in our case :/
def torch_quantize(module):
    module.register_forward_pre_hook(lambda self, input: torch.quantization.QuantStub()(input))
    module.register_forward_hook(lambda self, input, output: torch.quantization.DeQuantStub()(output))

    # module.train()
    # fused_model = torch.quantization.fuse_modules(module, [["conv1", "bn1", "relu"]], inplace=True)
    # for module_name, module in fused_model.named_children():
    #     if "layer" in module_name:
    #         for basic_block_name, basic_block in module.named_children():
    #             torch.quantization.fuse_modules(
    #                 basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True
    #             )
    #             for sub_block_name, sub_block in basic_block.named_children():
    #                 if sub_block_name == "downsample":
    #                     torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    qconfig_dict = torch.quantization.get_default_qat_qconfig("fbgemm")
    module.qconfig = qconfig_dict

    # module = torch.fx.symbolic_trace(module)  # TODO figure out a way to symbolic trace?
    # module = torch.quantization.prepare_fx(module, qconfig_dict)

    module = torch.quantization.prepare_qat(module)
    return module


def recursive_half(obj):
    if obj != None and not isinstance(obj, (str, float, int, list, dict, set)):
        for attr, val in obj.__dict__.items():
            if isinstance(val, (torch.Tensor, torch.nn.Parameter)):
                setattr(obj, attr, val.half())
            if isinstance(val, SynthesisBlock):
                val.use_fp16 = True
            recursive_half(getattr(obj, attr))


def print_and_half(x):
    print(x.flatten()[:10])
    return x.half()


def force_fp16(module):
    module.register_forward_pre_hook(lambda self, input: tuple(print_and_half(i) for i in input if i is not None))
    module = module.half()
    recursive_half(module)
    return module
