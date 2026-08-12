import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class KeyRegionGuidedFusion(nn.Module):
    """Lightweight key-region gate for neck fusion features.

    The module is designed to sit immediately after an Upsample+Concat fusion.
    It predicts spatial-channel gates from local and strip context, so the
    following C2f/C3k2 fusion block receives less background clutter and more
    behavior-relevant head/hand/book/phone cues.
    """

    def __init__(
        self,
        channels,
        reduction=4,
        init_scale=0.01,
        strip_kernel=7,
        context_mode="local_strip",
    ):
        super().__init__()
        if context_mode not in {"local", "strip", "local_strip"}:
            raise ValueError(f"Unsupported KRGFusion context mode: {context_mode}")
        self.context_mode = context_mode
        hidden_channels = max(16, channels // reduction)
        self.pre = ConvBNAct(channels, hidden_channels, 1)
        self.local_context = None
        if context_mode in {"local", "local_strip"}:
            self.local_context = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            )
        strip_padding = strip_kernel // 2
        self.horizontal_context = None
        self.vertical_context = None
        if context_mode in {"strip", "local_strip"}:
            self.horizontal_context = nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, strip_kernel),
                stride=1,
                padding=(0, strip_padding),
                groups=hidden_channels,
                bias=False,
            )
            self.vertical_context = nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(strip_kernel, 1),
                stride=1,
                padding=(strip_padding, 0),
                groups=hidden_channels,
                bias=False,
            )
        self.context_bn = nn.BatchNorm2d(hidden_channels)
        self.context_act = nn.SiLU(inplace=True)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        reduced = self.pre(x)
        context_parts = []
        if self.local_context is not None:
            context_parts.append(self.local_context(reduced))
        if self.horizontal_context is not None:
            context_parts.extend(
                (self.horizontal_context(reduced), self.vertical_context(reduced))
            )
        context = context_parts[0]
        for part in context_parts[1:]:
            context = context + part
        context = self.context_act(self.context_bn(context))

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((avg_map, max_map), dim=1))
        channel = self.channel_gate(context)
        gate = self.out_gate(context * channel) * spatial
        gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)
        modulation = gate.mul(2.0).sub(1.0)
        return x + torch.tanh(self.scale) * x * modulation


class KeyRegionFusionWrapper(nn.Module):
    """Wrap a YOLO Concat layer with key-region guided fusion."""

    def __init__(
        self,
        module,
        channels,
        reduction=4,
        init_scale=0.01,
        strip_kernel=7,
        context_mode="local_strip",
    ):
        super().__init__()
        self.module = module
        self.fusion = KeyRegionGuidedFusion(
            channels,
            reduction=reduction,
            init_scale=init_scale,
            strip_kernel=strip_kernel,
            context_mode=context_mode,
        )
        self.i = getattr(module, "i", None)
        self.f = getattr(module, "f", -1)
        self.c2 = channels
        self.out_channels = channels
        self.type = f"{getattr(module, 'type', module.__class__.__name__)}+KRGFusion"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x):
        return self.fusion(self.module(x))


def get_model_layers(yolo_model):
    if hasattr(yolo_model, "model") and isinstance(yolo_model.model, (nn.Sequential, nn.ModuleList)):
        return yolo_model.model
    if hasattr(yolo_model, "model") and hasattr(yolo_model.model, "model"):
        return yolo_model.model.model
    raise TypeError("Expected an Ultralytics YOLO wrapper or DetectionModel-like module.")


def is_concat_layer(module):
    module_name = module.__class__.__name__.lower()
    module_type = str(getattr(module, "type", "")).lower()
    return "concat" in module_name or "concat" in module_type


def infer_concat_output_channels(model_layers, layer_index):
    source_indices = getattr(model_layers[layer_index], "f", None)
    if not isinstance(source_indices, (list, tuple)):
        raise ValueError(f"Concat layer {layer_index} does not expose a source list in `.f`.")

    channels = 0
    for source_index in source_indices:
        resolved_index = layer_index + source_index if source_index < 0 else source_index
        channels += infer_layer_output_channels(model_layers, resolved_index)
    return channels


def infer_layer_output_channels(model_layers, layer_index, visited=None):
    """Infer output channels for graph layers, following pass-through layers."""
    if visited is None:
        visited = set()
    if layer_index in visited:
        raise ValueError(f"Cycle detected while inferring channels for layer {layer_index}.")
    visited.add(layer_index)

    module = model_layers[layer_index]
    if is_concat_layer(module):
        return infer_concat_output_channels(model_layers, layer_index)

    try:
        return infer_output_channels(module)
    except ValueError:
        source = getattr(module, "f", None)
        if isinstance(source, int):
            resolved_index = layer_index + source if source < 0 else source
            return infer_layer_output_channels(model_layers, resolved_index, visited)
        if isinstance(source, (list, tuple)) and len(source) == 1:
            resolved_index = layer_index + source[0] if source[0] < 0 else source[0]
            return infer_layer_output_channels(model_layers, resolved_index, visited)
        raise


def infer_output_channels(module):
    for attr in ("c2", "out_channels"):
        value = getattr(module, attr, None)
        if isinstance(value, int):
            return value

    for attr in ("cv3", "cv2", "cv1", "conv"):
        submodule = getattr(module, attr, None)
        if submodule is None:
            continue
        if isinstance(submodule, nn.Conv2d):
            return submodule.out_channels
        conv = getattr(submodule, "conv", None)
        if isinstance(conv, nn.Conv2d):
            return conv.out_channels

    convs = [submodule for submodule in module.modules() if isinstance(submodule, nn.Conv2d)]
    if convs:
        return convs[-1].out_channels
    raise ValueError(f"Unable to infer output channels for {module.__class__.__name__}.")


def resolve_fusion_layers(model_layers, target_layers=None, num_fusions=2):
    if target_layers is not None:
        return target_layers
    concat_layers = [
        index
        for index, module in enumerate(model_layers)
        if is_concat_layer(module)
    ]
    if not concat_layers:
        raise ValueError("Unable to find Concat layers for key-region guided fusion.")
    return concat_layers[:num_fusions]


def inject_key_region_guided_fusion(
    yolo_model,
    target_layers=None,
    num_fusions=2,
    reduction=4,
    init_scale=0.01,
    strip_kernel=7,
    context_mode="local_strip",
):
    model_layers = get_model_layers(yolo_model)
    target_layers = resolve_fusion_layers(
        model_layers,
        target_layers=target_layers,
        num_fusions=num_fusions,
    )

    injected = []
    for layer_index in target_layers:
        original = model_layers[layer_index]
        if isinstance(original, KeyRegionFusionWrapper):
            continue
        if not is_concat_layer(original):
            raise ValueError(
                f"Layer {layer_index} is not a Concat layer: {original.__class__.__name__}"
            )
        channels = infer_concat_output_channels(model_layers, layer_index)
        model_layers[layer_index] = KeyRegionFusionWrapper(
            original,
            channels=channels,
            reduction=reduction,
            init_scale=init_scale,
            strip_kernel=strip_kernel,
            context_mode=context_mode,
        )
        injected.append(
            {
                "layer": int(layer_index),
                "channels": int(channels),
                "reduction": int(reduction),
                "init_scale": float(init_scale),
                "strip_kernel": int(strip_kernel),
                "context_mode": context_mode,
            }
        )

    if not injected:
        print("Key-region guided fusion was already injected.")
    else:
        print(f"Injected key-region guided fusion: {injected}")
    return injected
