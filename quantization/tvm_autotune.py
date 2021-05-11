import functools
import os
from timeit import timeit as time

import dnnlib
import legacy
import onnx
import tvm
import tvm.contrib.graph_executor as runtime
from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner

import torch

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


def load_network():
    print("Loading...")
    with dnnlib.util.open_url(
        "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl"
    ) as fp:
        net_dict = legacy.load_network_pkl(fp)
        generator = net_dict["G"].requires_grad_(False).to(device)

    generator = generator.float()
    generator.forward = functools.partial(generator.forward, c=None, force_fp32=True)
    generator.eval()
    generator = torch.jit.trace(generator, torch.randn(input_shape, device=device)).eval()
    generator.to(device)
    for _ in range(5):
        generator(torch.randn(input_shape, device=device))  # warm up cudnn autotuner

    return generator


def randn(inputs, input_types):
    return tvm.relay.expr.const(
        torch.randn(
            size=tuple(int(i.data.asnumpy()) if isinstance(i, tvm.relay.Constant) else int(i) for i in inputs[0])
        ).numpy()
    )


def relay_module(generator, use_onnx=False):
    if use_onnx:
        torch.onnx.export(
            generator, torch.randn(input_shape, device=device), "generator.onnx", export_params=True, verbose=True
        )
        return relay.frontend.from_onnx(onnx.load("generator.onnx"), shape={"0": input_shape})
    return relay.frontend.from_pytorch(generator, [("input", input_shape)], {"aten::randn": randn})


def build(mod, params):
    print("Building...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=device, params=params)

    m = runtime.GraphModule(lib["default"](tvmdev))

    def generate(input):
        m.set_input("input", tvm.nd.array(input))
        m.run()
        return m.get_output(0)

    return generate


def autotune(mod, params):
    print("Tuning...")
    tasks = autotvm.task.extract_from_program(mod=mod, params=params, target=target)
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type="rank")

        if os.path.isfile(log_file):
            tuner_obj.load_history(autotvm.record.load_from_file(log_file))

        trials = min(tuning_trials, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=trials,
            early_stopping=tuning_early_stopping,
            measure_option=autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=30),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=15, min_repeat_ms=150),
            ),
            callbacks=[autotvm.callback.progress_bar(trials, prefix=prefix), autotvm.callback.log_to_file(log_file),],
        )


def tuned_generator():
    autotvm.record.pick_best(log_file, best_config)

    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        dev = tvm.device(str(target), 0)
        m = runtime.GraphModule(lib["default"](dev))

        def generate(input):
            m.set_input("input", tvm.nd.array(input))
            m.run()
            return m.get_output(0)

    return generate


if __name__ == "__main__":
    batch_size = 1
    input_shape = (batch_size, 512)
    output_shape = (batch_size, 3, 1024, 1024)

    device = "cuda"
    tvmdev = tvm.device(device)
    target = tvm.target.cuda()

    use_onnx = False

    tuning_trials = 2000
    tuning_early_stopping = 600

    network_name = "stylgan2-ada-generator"
    best_config = f"{network_name}.config"
    log_file = f"{network_name}.log"

    generator = load_network()

    mod, params = relay_module(generator, use_onnx=use_onnx)

    # tvmgen = build(mod, params)

    autotune(mod, params)
    tunegen = tuned_generator()

    print("PyTorch")
    print(time(lambda: generator(torch.randn(size=input_shape, device=device)), number=100) * 10, "ms")

    # print("ONNX TVM" if use_onnx else "TVM")
    # print(time(lambda: tvmgen(torch.randn(size=input_shape)), number=100) * 10, "ms")

    print("ONNX Tuned" if use_onnx else "Tuned")
    print(time(lambda: tunegen(torch.randn(size=input_shape)), number=100) * 10, "ms")
