import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2TwoStagePreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.extra.get("ltx2_two_stage_initialized", False):
            return batch

        final_height = int(batch.height)
        final_width = int(batch.width)
        stage_scale = int(
            getattr(server_args.pipeline_config, "two_stage_scale_factor", 2)
        )
        if stage_scale <= 0:
            raise ValueError(f"Invalid two_stage_scale_factor={stage_scale}.")
        if final_height % stage_scale != 0 or final_width % stage_scale != 0:
            raise ValueError(
                "LTX-2 two-stage generation requires final height/width to be divisible "
                f"by {stage_scale}, got height={final_height}, width={final_width}."
            )

        stage_1_height = final_height // stage_scale
        stage_1_width = final_width // stage_scale
        if stage_1_height % 32 != 0 or stage_1_width % 32 != 0:
            raise ValueError(
                "LTX-2 two-stage stage-1 dimensions must be divisible by 32, got "
                f"height={stage_1_height}, width={stage_1_width}."
            )

        batch.extra["ltx2_two_stage_initialized"] = True
        batch.extra["ltx2_stage_2_height"] = final_height
        batch.extra["ltx2_stage_2_width"] = final_width
        batch.height = stage_1_height
        batch.width = stage_1_width
        return batch


class LTX2LatentUpsampleStage(PipelineStage):
    def __init__(self, latent_upsampler):
        super().__init__()
        self.latent_upsampler = latent_upsampler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None:
            raise ValueError(
                "Expected stage-1 video latents before LTX-2 latent upsampling."
            )
        if not isinstance(batch.latents, torch.Tensor) or batch.latents.ndim != 5:
            raise ValueError(
                "LTX-2 latent upsampling expects unpacked video latents with shape [B, C, F, H, W], "
                f"got {type(batch.latents)} with shape {getattr(batch.latents, 'shape', None)}."
            )

        device = get_local_torch_device()
        latents = batch.latents.to(device=device)
        upsampler_dtype = next(self.latent_upsampler.parameters()).dtype
        self.latent_upsampler = self.latent_upsampler.to(
            device=device, dtype=upsampler_dtype
        )
        batch.latents = self.latent_upsampler(latents.to(dtype=upsampler_dtype))
        batch.height = int(batch.extra["ltx2_stage_2_height"])
        batch.width = int(batch.extra["ltx2_stage_2_width"])
        return batch


class LTX2RefinementLatentPreparationStage(PipelineStage):
    def __init__(self, vae, audio_vae):
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae

    @staticmethod
    def _create_noised_state(
        latents: torch.Tensor,
        noise_scale: float | torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        return noise_scale * noise + (1 - noise_scale) * latents

    @staticmethod
    def _normalize_video_latents(vae, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
        return (latents - latents_mean) * scaling_factor / latents_std

    @staticmethod
    def _normalize_audio_latents(audio_vae, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = audio_vae.latents_mean.to(
            device=latents.device, dtype=latents.dtype
        )
        latents_std = audio_vae.latents_std.to(
            device=latents.device, dtype=latents.dtype
        )
        if latents.ndim == 3:
            return (latents - latents_mean.view(1, 1, -1)) / latents_std.view(1, 1, -1)
        if latents.ndim == 2:
            return (latents - latents_mean.view(1, -1)) / latents_std.view(1, -1)
        return (latents - latents_mean) / latents_std

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is None:
            raise ValueError(
                "Expected stage-2 input video latents for LTX-2 refinement."
            )
        if not isinstance(batch.latents, torch.Tensor) or batch.latents.ndim != 5:
            raise ValueError(
                "LTX-2 refinement expects unpacked video latents with shape [B, C, F, H, W], "
                f"got {type(batch.latents)} with shape {getattr(batch.latents, 'shape', None)}."
            )

        sigma_values = getattr(
            server_args.pipeline_config, "stage_2_distilled_sigmas", None
        )
        if not sigma_values:
            raise ValueError("Missing `stage_2_distilled_sigmas` in pipeline config.")
        noise_scale = float(sigma_values[0])

        device = get_local_torch_device()
        dtype = batch.latents.dtype
        video_latents = batch.latents.to(device=device, dtype=dtype)
        normalized_video_latents = self._normalize_video_latents(
            self.vae, video_latents
        )

        conditioning_mask = torch.zeros(
            (
                normalized_video_latents.shape[0],
                1,
                normalized_video_latents.shape[2],
                normalized_video_latents.shape[3],
                normalized_video_latents.shape[4],
            ),
            device=device,
            dtype=normalized_video_latents.dtype,
        )
        conditioning_mask[:, :, 0] = 1.0
        noised_video_latents = self._create_noised_state(
            normalized_video_latents,
            noise_scale * (1 - conditioning_mask),
            generator=batch.generator,
        )

        packed_latents = server_args.pipeline_config.maybe_pack_latents(
            noised_video_latents, noised_video_latents.shape[0], batch
        )
        batch.latents = packed_latents
        batch.raw_latent_shape = packed_latents.shape
        batch.image_latent = None
        batch.ltx2_num_image_tokens = 0
        batch.condition_image = None

        if batch.audio_latents is not None:
            audio_latents = batch.audio_latents.to(device=device)
            audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
                audio_latents, audio_latents.shape[0], batch
            )
            audio_latents = self._normalize_audio_latents(self.audio_vae, audio_latents)
            batch.audio_latents = self._create_noised_state(
                audio_latents, noise_scale, generator=batch.generator
            )
            batch.raw_audio_latent_shape = batch.audio_latents.shape

        batch.do_classifier_free_guidance = False
        batch.guidance_scale = 1.0
        batch.sigmas = list(sigma_values)
        batch.timesteps = None
        batch.num_inference_steps = len(batch.sigmas) - 1
        batch.extra["ltx2_stage_2_noise_scale"] = noise_scale
        return batch


LTX2TwoStageSetupStage = LTX2TwoStagePreparationStage
