"""Standalone AMARU audio classifier extracted from the multimodal model.

The default ``AMARU`` configuration is the proposed architecture. The
``AMARU_TH*`` subclasses at the bottom of this file are the ablation variants:

    TH1: drop Block 5                         (4 blocks, 112-d embedding)
    TH2: drop Blocks 4 and 5                  (3 blocks,  40-d embedding)
    TH3: no adaptive branch selection         (T + F + L, equal-weight mean)
    TH4: no selection, no temporal branch     (F + L, equal-weight mean)
    TH5: no selection, no frequency branch    (T + L, equal-weight mean)
    TH6: no selection, no local branch        (T + F, equal-weight mean)

With the default arguments the module names (and therefore the state_dict
keys) are identical to the original AMARU, so existing checkpoints still load.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_backbone import BaseBackbone


ALL_BRANCHES = ("temporal", "frequency", "local")


def conv_norm(
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int],
    groups: int = 1,
    dilation: tuple[int, int] = (1, 1),
    activation: bool = True,
) -> nn.Sequential:
    """Build the convolution block used by the multimodal audio encoder."""
    padding = (
        (kernel[0] - 1) * dilation[0] // 2,
        (kernel[1] - 1) * dilation[1] // 2,
    )
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01, inplace=True) if activation else nn.Identity(),
    )


class MultiAxisResidualUnit(nn.Module):
    """Mix temporal, frequency, and local contexts with adaptive weights.

    Args:
        branches: subset of ``("temporal", "frequency", "local")`` to build.
        use_selector: if True, branches are mixed with softmax weights from the
            adaptive branch selector; if False, they are averaged with equal
            weights (1 / number of branches), which keeps the same output scale
            as the softmax mixture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        branches: Sequence[str] = ALL_BRANCHES,
        use_selector: bool = True,
    ) -> None:
        super().__init__()
        branches = tuple(branches)
        unknown = set(branches) - set(ALL_BRANCHES)
        if not branches or unknown or len(set(branches)) != len(branches):
            raise ValueError(f"branches must be a non-empty subset of {ALL_BRANCHES}, got {branches}.")
        # Keep the canonical order so the stacking order never depends on the caller.
        self.branches = tuple(name for name in ALL_BRANCHES if name in branches)
        self.use_selector = use_selector

        self.expand = conv_norm(in_channels, hidden_channels, (1, 1))
        if "temporal" in self.branches:
            self.temporal = nn.Sequential(
                conv_norm(hidden_channels, hidden_channels, (5, 1), groups=hidden_channels),
                conv_norm(
                    hidden_channels,
                    hidden_channels,
                    (3, 1),
                    groups=hidden_channels,
                    dilation=(2, 1),
                ),
            )
        if "frequency" in self.branches:
            self.frequency = nn.Sequential(
                conv_norm(hidden_channels, hidden_channels, (1, 5), groups=hidden_channels),
                conv_norm(
                    hidden_channels,
                    hidden_channels,
                    (1, 3),
                    groups=hidden_channels,
                    dilation=(1, 2),
                ),
            )
        if "local" in self.branches:
            self.local = conv_norm(
                hidden_channels,
                hidden_channels,
                (3, 3),
                groups=hidden_channels,
            )
        if self.use_selector:
            self.branch_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_channels, len(self.branches), kernel_size=1),
            )
        self.project = conv_norm(hidden_channels, out_channels, (1, 1), activation=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else conv_norm(in_channels, out_channels, (1, 1), activation=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        base = self.expand(x)
        branches = torch.stack(
            [getattr(self, name)(base) for name in self.branches],
            dim=1,
        )
        if self.use_selector:
            weights = F.softmax(self.branch_selector(base).flatten(1), dim=1)
            mixed = (branches * weights[:, :, None, None, None]).sum(dim=1)
        else:
            mixed = branches.mean(dim=1)
        return F.leaky_relu(
            self.project(mixed) + residual,
            negative_slope=0.01,
        )


class AudioBlock(nn.Sequential):
    """Two multi-axis residual units followed by 2x downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        branches: Sequence[str] = ALL_BRANCHES,
        use_selector: bool = True,
    ) -> None:
        super().__init__(
            MultiAxisResidualUnit(in_channels, out_channels, hidden_channels, branches, use_selector),
            MultiAxisResidualUnit(out_channels, out_channels, hidden_channels, branches, use_selector),
            nn.AvgPool2d(2),
        )


class AMARU(BaseBackbone):
    """Audio-only classifier using the multimodal branch's exact audio encoder.

    Args:
        classes_num: number of output classes.
        num_blocks: number of leading AudioBlocks to keep (1..5).
        branches: branches built inside every MultiAxisResidualUnit.
        use_selector: keep the adaptive branch selection (True) or use an
            equal-weight mean of the branches (False).
    """

    channels = (16, 24, 40, 112, 320)
    hidden_channels = (16, 24, 40, 64, 128)
    model_name_default = "amaru"

    def __init__(
        self,
        classes_num: int = 4,
        num_blocks: int = 5,
        branches: Sequence[str] = ALL_BRANCHES,
        use_selector: bool = True,
    ) -> None:
        super().__init__()
        if not 1 <= num_blocks <= len(self.channels):
            raise ValueError(f"num_blocks must be in [1, {len(self.channels)}], got {num_blocks}.")
        self.model_name = self.model_name_default
        self.num_blocks = num_blocks
        channels = self.channels[:num_blocks]
        hidden_channels = self.hidden_channels[:num_blocks]
        inputs = (1,) + channels[:-1]
        self.blocks = nn.ModuleList(
            AudioBlock(in_channels, out_channels, hidden, branches, use_selector)
            for in_channels, out_channels, hidden in zip(
                inputs,
                channels,
                hidden_channels,
            )
        )
        # This is the only layer added to the multimodal audio encoder so that
        # it satisfies the standalone BaseBackbone classification contract.
        # Keep this name for compatibility with extract_features.py.
        self.fc_audioset = nn.Linear(channels[-1], classes_num)

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=3)
        return x.amax(dim=2) + x.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.fc_audioset(self.pool(x))


# ---------------------------------------------------------------------------
# Ablation variants. Each has its own model_name so checkpoints and logs are
# written to checkpoint/<model_name>/ and never overwrite the full AMARU.
# ---------------------------------------------------------------------------


class AMARU_TH1(AMARU):
    """TH1: AMARU without Block 5."""

    model_name_default = "amaru_th1_no_block5"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, num_blocks=4)


class AMARU_TH2(AMARU):
    """TH2: AMARU without Blocks 4 and 5."""

    model_name_default = "amaru_th2_no_block45"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, num_blocks=3)


class AMARU_TH3(AMARU):
    """TH3: AMARU without adaptive branch selection (T + F + L averaged)."""

    model_name_default = "amaru_th3_no_selector"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, use_selector=False)


class AMARU_TH4(AMARU):
    """TH4: no adaptive branch selection, no temporal branch (F + L averaged)."""

    model_name_default = "amaru_th4_no_selector_no_temporal"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, branches=("frequency", "local"), use_selector=False)


class AMARU_TH5(AMARU):
    """TH5: no adaptive branch selection, no frequency branch (T + L averaged)."""

    model_name_default = "amaru_th5_no_selector_no_frequency"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, branches=("temporal", "local"), use_selector=False)


class AMARU_TH6(AMARU):
    """TH6: no adaptive branch selection, no local branch (T + F averaged)."""

    model_name_default = "amaru_th6_no_selector_no_local"

    def __init__(self, classes_num: int = 4) -> None:
        super().__init__(classes_num=classes_num, branches=("temporal", "frequency"), use_selector=False)
