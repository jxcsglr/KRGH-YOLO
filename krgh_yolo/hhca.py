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


class ClassificationKeyRegionAttention(nn.Module):
    """Task-aware key-region attention for the classification branch only."""

    def __init__(self, channels, reduction=4, init_scale=0.01, strip_kernel=7):
        super().__init__()
        hidden_channels = max(16, channels // reduction)
        self.pre = ConvBNAct(channels, hidden_channels, 1)
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
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        reduced = self.pre(x)
        context = (
            self.local_context(reduced)
            + self.horizontal_context(reduced)
            + self.vertical_context(reduced)
        )
        context = self.context_act(self.context_bn(context))
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((avg_map, max_map), dim=1))
        channel = self.channel_gate(context)
        gate = self.out_gate(context * channel) * spatial
        gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)
        modulation = gate.mul(2.0).sub(1.0)
        return x + torch.tanh(self.scale) * x * modulation


class CoordinateGate(nn.Module):
    """Coordinate gate that keeps vertical and horizontal positional cues."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = ConvBNAct(channels, channels, 1)
        self.height_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.width_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, height, width = x.shape
        height_context = x.mean(dim=3, keepdim=True)
        width_context = x.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat((height_context, width_context), dim=2)
        context = self.reduce(context)
        height_context, width_context = torch.split(context, [height, width], dim=2)
        width_context = width_context.transpose(2, 3)
        gate = torch.sigmoid(self.height_gate(height_context)) * torch.sigmoid(
            self.width_gate(width_context)
        )
        return torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)


class ClassificationHeadHandAttention(nn.Module):
    """Classification attention for head/hand-related fine-grained cues.

    This module is not a detector for heads or hands. It creates two lightweight
    cues inside the classification branch: strip/coordinate context for head and
    posture changes, and local depthwise detail for hands, phones, books, and
    pens. The output stays as a small residual gate on the original feature so
    box regression is not disturbed.
    """

    def __init__(
        self,
        channels,
        reduction=4,
        init_scale=0.01,
        strip_kernel=7,
        branch_mode="dual",
    ):
        super().__init__()
        if branch_mode not in {"posture", "hand_object", "dual"}:
            raise ValueError(f"Unsupported HHCA branch mode: {branch_mode}")
        self.branch_mode = branch_mode
        hidden_channels = max(16, channels // reduction)
        branch_count = 2 if branch_mode == "dual" else 1
        self.pre = ConvBNAct(channels, hidden_channels * branch_count, 1)
        strip_padding = strip_kernel // 2
        self.head_context = None
        if branch_mode in {"posture", "dual"}:
            self.head_context = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(strip_kernel, 1),
                    stride=1,
                    padding=(strip_padding, 0),
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(1, strip_kernel),
                    stride=1,
                    padding=(0, strip_padding),
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            )
        self.hand_context = None
        if branch_mode in {"hand_object", "dual"}:
            self.hand_context = nn.Sequential(
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
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            )
        self.context_fuse = ConvBNAct(hidden_channels * branch_count, hidden_channels, 1)
        self.coordinate_gate = CoordinateGate(hidden_channels)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Sigmoid(),
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        projected = self.pre(x)
        if self.branch_mode == "dual":
            head_source, hand_source = projected.chunk(2, dim=1)
            contexts = (
                self.head_context(head_source),
                self.hand_context(hand_source),
            )
        elif self.branch_mode == "posture":
            contexts = (self.head_context(projected),)
        else:
            contexts = (self.hand_context(projected),)
        context = self.context_fuse(torch.cat(contexts, dim=1))
        context = context * self.coordinate_gate(context) * self.channel_gate(context)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((avg_map, max_map), dim=1))
        gate = self.out_gate(context) * spatial
        gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)
        modulation = gate.mul(2.0).sub(1.0)
        return x + torch.tanh(self.scale) * x * modulation


class ClassificationCBAMAttention(nn.Module):
    """CBAM control for the Detect classification branch.

    The channel-then-spatial attention and direct feature gating follow the
    canonical CBAM formulation. This module is intentionally not residualized
    so the experiment remains a recognizable off-the-shelf CBAM control.
    """

    def __init__(self, channels, reduction=4, init_scale=0.01, strip_kernel=7):
        super().__init__()
        del init_scale, strip_kernel  # Kept in the shared constructor contract.
        hidden_channels = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_channel = self.channel_mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_channel = self.channel_mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        channel_gate = torch.sigmoid(avg_channel + max_channel)
        channel_refined = x * channel_gate

        avg_spatial = channel_refined.mean(dim=1, keepdim=True)
        max_spatial = channel_refined.amax(dim=1, keepdim=True)
        spatial_gate = self.spatial_gate(torch.cat((avg_spatial, max_spatial), dim=1))
        return channel_refined * spatial_gate


class ClassificationBranchWrapper(nn.Module):
    """Prepend task-aware attention to a Detect classification branch."""

    def __init__(
        self,
        module,
        channels,
        reduction=4,
        init_scale=0.01,
        strip_kernel=7,
        attention_kind="key_region",
        branch_mode="dual",
    ):
        super().__init__()
        attention_modules = {
            "key_region": ClassificationKeyRegionAttention,
            "head_hand": ClassificationHeadHandAttention,
            "cbam": ClassificationCBAMAttention,
        }
        if attention_kind not in attention_modules:
            raise ValueError(f"Unsupported classification attention kind: {attention_kind}")
        self.attention_kind = attention_kind
        attention_kwargs = {
            "reduction": reduction,
            "init_scale": init_scale,
            "strip_kernel": strip_kernel,
        }
        if attention_kind == "head_hand":
            attention_kwargs["branch_mode"] = branch_mode
        self.attention = attention_modules[attention_kind](channels, **attention_kwargs)
        self.module = module

    def forward(self, x):
        return self.module(self.attention(x))


def get_model_layers(yolo_model):
    if hasattr(yolo_model, "model") and isinstance(yolo_model.model, (nn.Sequential, nn.ModuleList)):
        return yolo_model.model
    if hasattr(yolo_model, "model") and hasattr(yolo_model.model, "model"):
        return yolo_model.model.model
    raise TypeError("Expected an Ultralytics YOLO wrapper or DetectionModel-like module.")


def get_detect_layer(yolo_model):
    model_layers = get_model_layers(yolo_model)
    detect_layer = model_layers[-1]
    if not hasattr(detect_layer, "cv3"):
        raise ValueError("The final model layer does not look like an Ultralytics Detect layer.")
    return detect_layer


def infer_input_channels(module):
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            return submodule.in_channels
        conv = getattr(submodule, "conv", None)
        if isinstance(conv, nn.Conv2d):
            return conv.in_channels
    raise ValueError(f"Unable to infer input channels for {module.__class__.__name__}.")


def resolve_target_scales(detect_layer, target_scales=None, num_scales=2):
    scale_count = len(getattr(detect_layer, "cv3", []))
    if target_scales is None:
        return list(range(min(num_scales, scale_count)))
    return target_scales


def inject_classification_attention_head(
    yolo_model,
    target_scales=None,
    num_scales=2,
    reduction=4,
    init_scale=0.01,
    strip_kernel=7,
    attention_kind="key_region",
    branch_mode="dual",
):
    detect_layer = get_detect_layer(yolo_model)
    target_scales = resolve_target_scales(
        detect_layer,
        target_scales=target_scales,
        num_scales=num_scales,
    )

    injected = []
    for scale_index in target_scales:
        if scale_index < 0 or scale_index >= len(detect_layer.cv3):
            raise ValueError(f"Detect scale index out of range: {scale_index}")
        original = detect_layer.cv3[scale_index]
        if isinstance(original, ClassificationBranchWrapper):
            continue
        channels = infer_input_channels(original)
        detect_layer.cv3[scale_index] = ClassificationBranchWrapper(
            original,
            channels=channels,
            reduction=reduction,
            init_scale=init_scale,
            strip_kernel=strip_kernel,
            attention_kind=attention_kind,
            branch_mode=branch_mode,
        )
        record = {
            "scale": int(scale_index),
            "channels": int(channels),
            "reduction": int(reduction),
            "attention_kind": attention_kind,
        }
        if attention_kind != "cbam":
            record.update(
                {
                    "init_scale": float(init_scale),
                    "strip_kernel": int(strip_kernel),
                }
            )
        if attention_kind == "head_hand":
            record["branch_mode"] = branch_mode
        injected.append(record)

    if not injected:
        print(f"Classification {attention_kind} head was already injected.")
    else:
        print(f"Injected classification {attention_kind} head: {injected}")
    return injected


def inject_classification_key_region_head(
    yolo_model,
    target_scales=None,
    num_scales=2,
    reduction=4,
    init_scale=0.01,
    strip_kernel=7,
):
    return inject_classification_attention_head(
        yolo_model,
        target_scales=target_scales,
        num_scales=num_scales,
        reduction=reduction,
        init_scale=init_scale,
        strip_kernel=strip_kernel,
        attention_kind="key_region",
    )


def inject_classification_head_hand_attention(
    yolo_model,
    target_scales=None,
    num_scales=2,
    reduction=4,
    init_scale=0.01,
    strip_kernel=7,
    branch_mode="dual",
):
    return inject_classification_attention_head(
        yolo_model,
        target_scales=target_scales,
        num_scales=num_scales,
        reduction=reduction,
        init_scale=init_scale,
        strip_kernel=strip_kernel,
        attention_kind="head_hand",
        branch_mode=branch_mode,
    )


def inject_classification_cbam_attention(
    yolo_model,
    target_scales=None,
    num_scales=2,
    reduction=4,
    init_scale=0.01,
):
    return inject_classification_attention_head(
        yolo_model,
        target_scales=target_scales,
        num_scales=num_scales,
        reduction=reduction,
        init_scale=init_scale,
        attention_kind="cbam",
    )
