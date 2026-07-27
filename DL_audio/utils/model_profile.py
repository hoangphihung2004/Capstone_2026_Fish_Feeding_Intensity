import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    parameters: int
    flops: int
    gflops: float


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_flops(model: nn.Module, example_input: torch.Tensor) -> int:
    """Count FLOPs using the same forward-hook method as the original U-FFIA code."""
    multiply_adds = True
    handles = []

    list_conv2d = []

    def conv2d_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch_size, _, _, _ = inputs[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if module.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        list_conv2d.append(batch_size * params * output_height * output_width)

    list_conv1d = []

    def conv1d_hook(module: nn.Conv1d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch_size, _, _ = inputs[0].size()
        output_channels, output_length = output[0].size()
        kernel_ops = module.kernel_size[0] * (module.in_channels / module.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if module.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops)
        list_conv1d.append(batch_size * params * output_length)

    list_linear = []

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch_size = inputs[0].size(0) if inputs[0].dim() == 2 else 1
        weight_ops = module.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = module.bias.nelement() if module.bias is not None else 0
        list_linear.append(batch_size * (weight_ops + bias_ops))

    list_bn = []

    def bn_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        list_bn.append(inputs[0].nelement() * 2)

    list_relu = []

    def relu_hook(module: nn.ReLU, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        list_relu.append(inputs[0].nelement() * 2)

    list_pooling2d = []

    def pooling2d_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch_size, _, _, _ = inputs[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        kernel_ops = kernel_size * kernel_size
        params = output_channels * kernel_ops
        list_pooling2d.append(batch_size * params * output_height * output_width)

    list_pooling1d = []

    def pooling1d_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        batch_size, _, _ = inputs[0].size()
        output_channels, output_length = output[0].size()
        kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        params = output_channels * kernel_size
        list_pooling1d.append(batch_size * params * output_length)

    def register_hooks(net: nn.Module) -> None:
        children = list(net.children())
        if children:
            for child in children:
                register_hooks(child)
            return

        if isinstance(net, nn.Conv2d):
            handles.append(net.register_forward_hook(conv2d_hook))
        elif isinstance(net, nn.Conv1d):
            handles.append(net.register_forward_hook(conv1d_hook))
        elif isinstance(net, nn.Linear):
            handles.append(net.register_forward_hook(linear_hook))
        elif isinstance(net, (nn.BatchNorm2d, nn.BatchNorm1d)):
            handles.append(net.register_forward_hook(bn_hook))
        elif isinstance(net, nn.ReLU):
            handles.append(net.register_forward_hook(relu_hook))
        elif isinstance(net, (nn.AvgPool2d, nn.MaxPool2d)):
            handles.append(net.register_forward_hook(pooling2d_hook))
        elif isinstance(net, (nn.AvgPool1d, nn.MaxPool1d)):
            handles.append(net.register_forward_hook(pooling1d_hook))

    register_hooks(model)

    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for handle in handles:
            handle.remove()

    total_flops = sum(list_conv2d) + sum(list_conv1d) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling2d) + sum(list_pooling1d)
    return int(total_flops)


def profile_model(model: nn.Module, example_input: torch.Tensor) -> ModelProfile:
    was_training = model.training
    model.eval()

    try:
        parameters = count_parameters(model)
        flops = estimate_flops(model, example_input)
    finally:
        model.train(was_training)

    return ModelProfile(
        parameters=parameters,
        flops=flops,
        gflops=flops / 1e9,
    )


def log_model_profile(model: nn.Module, example_input: torch.Tensor, model_name: str = "model") -> ModelProfile:
    profile = profile_model(model=model, example_input=example_input)

    logger.info("==================================================")
    logger.info(f"Model Profile: {model_name}")
    logger.info(f"  - Number of Parameters:      {profile.parameters:,}")
    logger.info(f"  - FLOPs / Clip:              {profile.flops:,}")
    logger.info(f"  - GFLOPs / Clip:             {profile.gflops:.4f}")
    logger.info("==================================================")

    return profile
