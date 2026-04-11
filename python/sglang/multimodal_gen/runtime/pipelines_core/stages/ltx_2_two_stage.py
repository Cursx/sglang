import os

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2Stage2LoRAControlStage(PipelineStage):
    def __init__(
        self,
        pipeline,
        enable: bool,
        lora_path: str | None = None,
        lora_nickname: str | None = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.enable = enable
        self.lora_path = lora_path
        self.lora_nickname = lora_nickname or "stage_2_distilled"

    def _validate_lora_path(self) -> None:
        if self.lora_path is None or not os.path.exists(self.lora_path):
            raise ValueError(
                "LTX-2 two-stage refinement requires the distilled LoRA weights, "
                f"but the resolved path does not exist: {self.lora_path!r}."
            )

    def _get_preload_flag_name(self) -> str:
        return f"_ltx2_stage2_lora_preloaded_{self.lora_nickname}"

    def _is_preloaded(self) -> bool:
        return bool(getattr(self.pipeline, self._get_preload_flag_name(), False))

    def _mark_preloaded(self) -> None:
        setattr(self.pipeline, self._get_preload_flag_name(), True)

    def _reset_runtime_attention_cache(self) -> None:
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is not None and hasattr(
            transformer, "reset_runtime_attention_cache"
        ):
            transformer.reset_runtime_attention_cache()

    @staticmethod
    def _synchronize_device() -> None:
        device_module = torch.get_device_module()
        if hasattr(device_module, "is_available") and device_module.is_available():
            device_module.synchronize()

    def _ensure_preloaded(self) -> None:
        if self._is_preloaded():
            return

        self._validate_lora_path()
        self.pipeline.set_lora(
            self.lora_nickname,
            self.lora_path,
            target="transformer",
            strength=1.0,
            merge_weights=False,
        )
        self._mark_preloaded()

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not (
            hasattr(self.pipeline, "set_lora")
            and hasattr(self.pipeline, "deactivate_lora_weights")
        ):
            raise TypeError(
                "LTX2 stage-2 LoRA control requires an LoRA-capable pipeline."
            )

        if not self.enable:
            if server_args.lora_path is not None:
                self.pipeline.set_lora(
                    server_args.lora_nickname,
                    server_args.lora_path,
                    target="transformer",
                    strength=float(server_args.lora_scale),
                )
            else:
                self._ensure_preloaded()
                self.pipeline.deactivate_lora_weights(target="transformer")
            self._synchronize_device()
            self._reset_runtime_attention_cache()
            return batch

        if server_args.lora_path is not None:
            self._validate_lora_path()
            self.pipeline.set_lora(
                [server_args.lora_nickname, self.lora_nickname],
                [server_args.lora_path, self.lora_path],
                target=["transformer", "transformer"],
                strength=[float(server_args.lora_scale), 1.0],
                merge_weights=True,
            )
        else:
            self._ensure_preloaded()
            self.pipeline.set_lora(
                self.lora_nickname,
                target="transformer",
                strength=1.0,
                merge_weights=False,
            )
        self._synchronize_device()
        self._reset_runtime_attention_cache()

        return batch


class LTX2Stage2LoRADisableStage(LTX2Stage2LoRAControlStage):
    def __init__(
        self,
        pipeline,
        lora_path: str | None = None,
        lora_nickname: str | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            enable=False,
            lora_path=lora_path,
            lora_nickname=lora_nickname,
        )


class LTX2Stage2LoRAEnableStage(LTX2Stage2LoRAControlStage):
    def __init__(
        self,
        pipeline,
        lora_path: str | None = None,
        lora_nickname: str | None = None,
    ):
        super().__init__(
            pipeline=pipeline,
            enable=True,
            lora_path=lora_path,
            lora_nickname=lora_nickname,
        )


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
    def __init__(self, latent_upsampler, vae):
        super().__init__()
        self.latent_upsampler = latent_upsampler
        self.vae = vae

    @staticmethod
    def _denormalize_video_latents(vae, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
        return latents * latents_std / scaling_factor + latents_mean

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
        latents = self._denormalize_video_latents(self.vae, latents)
        upsampler_dtype = next(self.latent_upsampler.parameters()).dtype
        self.latent_upsampler = self.latent_upsampler.to(
            device=device, dtype=upsampler_dtype
        )
        batch.latents = self.latent_upsampler(latents.to(dtype=upsampler_dtype))
        batch.height = int(batch.extra["ltx2_stage_2_height"])
        batch.width = int(batch.extra["ltx2_stage_2_width"])
        return batch


class LTX2RefinementLatentPreparationStage(PipelineStage):
    def __init__(self, vae, audio_vae, preserve_conditioned_first_frame: bool):
        super().__init__()
        self.vae = vae
        self.audio_vae = audio_vae
        self.preserve_conditioned_first_frame = preserve_conditioned_first_frame

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

    @staticmethod
    def _reset_stage2_generators(batch: Req) -> None:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list) and generator:
            generator_device = str(generator[0].device)
        elif isinstance(generator, torch.Generator):
            generator_device = str(generator.device)
        else:
            generator_device = "cpu"

        seeds = getattr(batch, "seeds", None)
        if not seeds:
            seed = getattr(batch, "seed", None)
            if seed is None:
                return
            seeds = [int(seed)]

        batch.generator = [
            torch.Generator(device=generator_device).manual_seed(int(seed))
            for seed in seeds
        ]

    @staticmethod
    def _has_image_conditioning(batch: Req) -> bool:
        if batch.image_path is not None:
            return True
        if batch.condition_image is not None:
            return True
        if batch.image_latent is not None:
            return True
        return int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0

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
        batch.extra["ltx2_phase"] = "stage2"
        self._reset_stage2_generators(batch)

        device = get_local_torch_device()
        dtype = batch.latents.dtype
        video_latents = batch.latents.to(device=device, dtype=dtype)
        normalized_video_latents = self._normalize_video_latents(
            self.vae, video_latents
        )
        packed_video_latents = server_args.pipeline_config.maybe_pack_latents(
            normalized_video_latents, normalized_video_latents.shape[0], batch
        )
        noised_video_latents = self._create_noised_state(
            packed_video_latents,
            noise_scale,
            generator=batch.generator,
        )
        if (
            self.preserve_conditioned_first_frame
            and self._has_image_conditioning(batch)
            and batch.image_path is None
        ):
            seq_len = int(packed_video_latents.shape[1])
            preserved_tokens = int(getattr(batch, "ltx2_num_image_tokens", 0))
            inferred_tokens_per_frame = None
            if batch.debug or preserved_tokens <= 0 or preserved_tokens > seq_len:
                _, inferred_tokens_per_frame = (
                    server_args.pipeline_config._infer_video_latent_frames_and_tokens_per_frame(
                        batch, seq_len
                    )
                )
            if preserved_tokens <= 0 or preserved_tokens > seq_len:
                preserved_tokens = int(inferred_tokens_per_frame)
            elif (
                batch.debug
                and inferred_tokens_per_frame is not None
                and int(inferred_tokens_per_frame) != preserved_tokens
            ):
                logger.info(
                    "LTX-2 stage2 preserving %d image tokens; inferred tokens_per_frame=%d",
                    preserved_tokens,
                    int(inferred_tokens_per_frame),
                )
            noised_video_latents[:, :preserved_tokens, :] = packed_video_latents[
                :, :preserved_tokens, :
            ]

        batch.latents = noised_video_latents
        batch.raw_latent_shape = noised_video_latents.shape
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
        batch.num_inference_steps = len(batch.sigmas)
        batch.extra["ltx2_stage_2_noise_scale"] = noise_scale
        return batch


LTX2TwoStageSetupStage = LTX2TwoStagePreparationStage


def build_ltx2_two_stage_pipeline_cls(base_cls, pipeline_config_cls):
    class LTX2TwoStagePipeline(base_cls):
        pipeline_name = "LTX2TwoStagePipeline"

        def create_pipeline_stages(self, server_args: ServerArgs):
            self._create_two_stage_pipeline_stages(
                server_args, preserve_conditioned_first_frame=False
            )

    LTX2TwoStagePipeline.pipeline_config_cls = pipeline_config_cls
    LTX2TwoStagePipeline.__module__ = __name__
    return LTX2TwoStagePipeline
