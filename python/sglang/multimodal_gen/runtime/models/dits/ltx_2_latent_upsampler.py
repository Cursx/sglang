# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RATIONAL_RESAMPLER_SCALE_MAPPING = {
    0.75: (3, 4),
    1.5: (3, 2),
    2.0: (2, 1),
    4.0: (4, 1),
}


class ResBlock(nn.Module):
    def __init__(self, channels: int, mid_channels: int | None = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv_cls = nn.Conv2d if dims == 2 else nn.Conv3d
        self.conv1 = conv_cls(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.conv2 = conv_cls(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation(hidden_states + residual)
        return hidden_states


class PixelShuffleND(nn.Module):
    def __init__(self, dims: int, upscale_factors=(2, 2, 2)):
        super().__init__()
        if dims not in (1, 2, 3):
            raise ValueError(f"Unsupported dims={dims}. Expected 1, 2, or 3.")
        self.dims = int(dims)
        self.upscale_factors = tuple(upscale_factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:3]))
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        if self.dims == 2:
            return (
                x.unflatten(1, (-1, *self.upscale_factors[:2]))
                .permute(0, 1, 4, 2, 5, 3)
                .flatten(4, 5)
                .flatten(2, 3)
            )
        return (
            x.unflatten(1, (-1, *self.upscale_factors[:1]))
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(2, 3)
        )


class BlurDownsample(nn.Module):
    def __init__(self, dims: int, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f"`dims` must be 2 or 3, got {dims}.")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError(
                f"`kernel_size` must be an odd integer >= 3, got {kernel_size}."
            )

        self.dims = int(dims)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)

        coeffs = torch.tensor(
            [math.comb(kernel_size - 1, k) for k in range(kernel_size)],
            dtype=torch.float32,
        )
        kernel_2d = coeffs[:, None] @ coeffs[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        if self.dims == 2:
            channels = x.shape[1]
            weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
            return F.conv2d(
                x,
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.kernel_size // 2,
                groups=channels,
            )

        batch_size, channels, num_frames, _, _ = x.shape
        x = x.transpose(1, 2).flatten(0, 1)
        weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(
            x,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=channels,
        )
        h2, w2 = x.shape[-2:]
        return x.unflatten(0, (batch_size, num_frames)).reshape(
            batch_size, -1, num_frames, h2, w2
        )


class SpatialRationalResampler(nn.Module):
    def __init__(self, mid_channels: int = 1024, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)
        num_denom = RATIONAL_RESAMPLER_SCALE_MAPPING.get(self.scale)
        if num_denom is None:
            raise ValueError(
                f"Unsupported spatial upsample scale {self.scale}. "
                f"Supported values: {list(RATIONAL_RESAMPLER_SCALE_MAPPING.keys())}"
            )
        self.num, self.den = num_denom
        self.conv = nn.Conv2d(
            mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        return x


class LTX2LatentUpsamplerModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        rational_spatial_scale: float = 2.0,
        use_rational_resampler: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.mid_channels = int(mid_channels)
        self.num_blocks_per_stage = int(num_blocks_per_stage)
        self.dims = int(dims)
        self.spatial_upsample = bool(spatial_upsample)
        self.temporal_upsample = bool(temporal_upsample)

        conv_cls = nn.Conv2d if self.dims == 2 else nn.Conv3d
        self.initial_conv = conv_cls(
            self.in_channels, self.mid_channels, kernel_size=3, padding=1
        )
        self.initial_norm = nn.GroupNorm(32, self.mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [
                ResBlock(self.mid_channels, dims=self.dims)
                for _ in range(self.num_blocks_per_stage)
            ]
        )

        if self.spatial_upsample and self.temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(
                    self.mid_channels, 8 * self.mid_channels, kernel_size=3, padding=1
                ),
                PixelShuffleND(3),
            )
        elif self.spatial_upsample:
            if use_rational_resampler:
                self.upsampler = SpatialRationalResampler(
                    mid_channels=self.mid_channels, scale=rational_spatial_scale
                )
            else:
                self.upsampler = nn.Sequential(
                    nn.Conv2d(
                        self.mid_channels,
                        4 * self.mid_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    PixelShuffleND(2),
                )
        elif self.temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(
                    self.mid_channels, 2 * self.mid_channels, kernel_size=3, padding=1
                ),
                PixelShuffleND(1),
            )
        else:
            raise ValueError(
                "At least one of spatial_upsample or temporal_upsample must be True."
            )

        self.post_upsample_res_blocks = nn.ModuleList(
            [
                ResBlock(self.mid_channels, dims=self.dims)
                for _ in range(self.num_blocks_per_stage)
            ]
        )
        self.final_conv = conv_cls(
            self.mid_channels, self.in_channels, kernel_size=3, padding=1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        if self.dims == 2:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)
            for block in self.res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.upsampler(hidden_states)
            for block in self.post_upsample_res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.final_conv(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )
            return hidden_states

        hidden_states = self.initial_conv(hidden_states)
        hidden_states = self.initial_norm(hidden_states)
        hidden_states = self.initial_activation(hidden_states)
        for block in self.res_blocks:
            hidden_states = block(hidden_states)

        if self.temporal_upsample:
            hidden_states = self.upsampler(hidden_states)
            hidden_states = hidden_states[:, :, 1:, :, :]
        else:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.upsampler(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(
                0, 2, 1, 3, 4
            )

        for block in self.post_upsample_res_blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_conv(hidden_states)
        return hidden_states


EntryClass = LTX2LatentUpsamplerModel
