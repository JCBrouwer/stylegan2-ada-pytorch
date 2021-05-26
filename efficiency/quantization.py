import copy
from abc import abstractmethod

import torch

_NBITS = 8
_ACTMAX = 4.0


class LinQuantSteOp(torch.autograd.Function):
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


quantize = LinQuantSteOp.apply


class Quantization:
    @abstractmethod
    def forward_pre_hook(self, module, input):
        pass


class Linear(Quantization):
    def __init__(self, parent, input_quant, input_signed):
        self.parent = parent
        self.input_quant = input_quant
        self.input_signed = input_signed

        self.nbits = _NBITS
        self.input_max = _ACTMAX

        print("\nQuantizing layers")
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if "fc" in name or "affine" in name or "conv" in name:
                mod.register_forward_pre_hook(self.forward_pre_hook)
                print(name, mod.weight.shape)
        print()

    def forward_pre_hook(self, module, input):
        if hasattr(module, "weight"):
            module.weight.data = quantize(module.weight, True, self.nbits, module.weight.abs().max().item())

        if self.input_quant:
            output = []
            for i in range(len(input)):
                max_val = self.input_max
                min_val = -max_val if self.input_signed else 0.0
                output.append(
                    quantize(input[i].clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val)
                )
            return tuple(output)


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
