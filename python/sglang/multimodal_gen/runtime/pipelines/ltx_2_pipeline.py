import math
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2ImageToVideoPipelineConfig,
    LTX2ImageToVideoTwoStagesPipelineConfig,
    LTX2PipelineConfig,
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    LTX2AVDecodingStage,
    LTX2AVDenoisingStage,
    LTX2AVLatentPreparationStage,
    LTX2LatentUpsampleStage,
    LTX2RefinementLatentPreparationStage,
    LTX2RefinementStage,
    LTX2Stage2LoRADisableStage,
    LTX2Stage2LoRAEnableStage,
    LTX2TextConnectorStage,
    LTX2TwoStagePreparationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_two_stage import (
    build_ltx2_two_stage_pipeline_cls,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


def _resolve_ltx2_two_stage_component_paths(
    model_path: str, component_paths: dict[str, str]
) -> dict[str, str]:
    resolved = dict(component_paths)
    auto_resolved = []

    if "spatial_upsampler" not in resolved and "latent_upsampler" in resolved:
        resolved["spatial_upsampler"] = resolved["latent_upsampler"]
    elif "latent_upsampler" not in resolved and "spatial_upsampler" in resolved:
        resolved["latent_upsampler"] = resolved["spatial_upsampler"]

    if "spatial_upsampler" not in resolved:
        spatial_candidates = [
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            os.path.join(model_path, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            os.path.join(model_path, "latent_upsampler"),
            os.path.join(model_path, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ]
        for candidate in spatial_candidates:
            if os.path.exists(candidate):
                resolved["spatial_upsampler"] = candidate
                resolved["latent_upsampler"] = candidate
                auto_resolved.append(f"spatial_upsampler={candidate}")
                break

    if "distilled_lora" not in resolved and "stage_2_distilled_lora" in resolved:
        resolved["distilled_lora"] = resolved["stage_2_distilled_lora"]
    elif "stage_2_distilled_lora" not in resolved and "distilled_lora" in resolved:
        resolved["stage_2_distilled_lora"] = resolved["distilled_lora"]

    if "distilled_lora" not in resolved:
        distilled_lora_candidates = [
            os.path.join(model_path, "ltx-2.3-20b-distilled-lora-384.safetensors"),
            os.path.join(model_path, "ltx-2.3-22b-distilled-lora-384.safetensors"),
            os.path.join(model_path, "ltx-2-19b-distilled-lora-384.safetensors"),
        ]
        for distilled_lora in distilled_lora_candidates:
            if os.path.exists(distilled_lora):
                resolved["distilled_lora"] = distilled_lora
                resolved["stage_2_distilled_lora"] = distilled_lora
                auto_resolved.append(f"distilled_lora={distilled_lora}")
                break

    if auto_resolved:
        logger.info(
            "Auto-resolved LTX2 two-stage components: %s", ", ".join(auto_resolved)
        )

    return resolved


def calculate_ltx2_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_SHIFT_ANCHOR,
    max_seq_len: int = MAX_SHIFT_ANCHOR,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    mm = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - mm * base_seq_len
    return image_seq_len * mm + b


def prepare_ltx2_mu(batch: Req, server_args: ServerArgs):
    if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
        return "mu", None
    latent_num_frames = (int(batch.num_frames) - 1) // int(
        server_args.pipeline_config.vae_temporal_compression
    ) + 1
    latent_height = int(batch.height) // int(
        server_args.pipeline_config.vae_scale_factor
    )
    latent_width = int(batch.width) // int(server_args.pipeline_config.vae_scale_factor)
    video_sequence_length = latent_num_frames * latent_height * latent_width
    return "mu", calculate_ltx2_shift(video_sequence_length)


def build_official_ltx2_sigmas(
    steps: int,
    *,
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    stretch: bool = True,
    terminal: float = 0.1,
    default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
) -> list[float]:
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
    b = base_shift - mm * BASE_SHIFT_ANCHOR
    sigma_shift = float(default_number_of_tokens) * mm + b

    non_zero_mask = sigmas != 0
    shifted = torch.where(
        non_zero_mask,
        math.exp(sigma_shift) / (math.exp(sigma_shift) + (1.0 / sigmas - 1.0)),
        torch.zeros_like(sigmas),
    )

    if stretch:
        one_minus_z = 1.0 - shifted[non_zero_mask]
        scale_factor = one_minus_z[-1] / (1.0 - terminal)
        shifted[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

    return shifted[:-1].tolist()


class LTX2SigmaPreparationStage(PipelineStage):
    """Prepare native LTX-2 sigma schedule before timestep setup."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"
        if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
            batch.sigmas = build_official_ltx2_sigmas(int(batch.num_inference_steps))
        else:
            batch.sigmas = np.linspace(
                1.0,
                1.0 / int(batch.num_inference_steps),
                int(batch.num_inference_steps),
            ).tolist()
        return batch


class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        if sigmas is not None and timesteps is None and mu is None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self.num_inference_steps = len(timesteps)
            self.timesteps = timesteps
            self.sigmas = sigmas
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class LTX2BasePipeline(LoRAPipeline, ComposedPipelineBase):
    is_video_pipeline = True
    pipeline_config_cls = LTX2PipelineConfig
    sampling_params_cls = LTX2SamplingParams

    _required_config_modules = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        orig = self.get_module("scheduler")
        self.modules["scheduler"] = LTX2FlowMatchScheduler.from_config(orig.config)

    def _add_shared_preprocessing_stages(self):
        self.add_stages(
            [
                TextEncodingStage(
                    text_encoders=[self.get_module("text_encoder")],
                    tokenizers=[self.get_module("tokenizer")],
                ),
                LTX2TextConnectorStage(connectors=self.get_module("connectors")),
                LTX2SigmaPreparationStage(),
            ]
        )
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_ltx2_mu]
        )

    def _add_stage_1_denoising(self):
        self.add_stages(
            [
                LTX2AVLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                LTX2AVDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    pipeline=self,
                ),
            ]
        )

    def _add_decoding_stage(self):
        self.add_stage(
            LTX2AVDecodingStage(
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
                pipeline=self,
            ),
        )


class LTX2Pipeline(LTX2BasePipeline):
    pipeline_name = "LTX2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self._add_shared_preprocessing_stages()
        self._add_stage_1_denoising()
        self._add_decoding_stage()


class LTX2ImageToVideoPipeline(LTX2Pipeline):
    pipeline_name = "LTX2ImageToVideoPipeline"
    pipeline_config_cls = LTX2ImageToVideoPipelineConfig


class LTX2ImageToVideoTwoStagesPipeline(LTX2BasePipeline):
    pipeline_name = "LTX2ImageToVideoTwoStagesPipeline"
    pipeline_config_cls = LTX2ImageToVideoTwoStagesPipelineConfig

    def _resolve_stage_2_lora_path(self, server_args: ServerArgs) -> str:
        override_path = server_args.component_paths.get(
            "stage_2_distilled_lora"
        ) or server_args.component_paths.get("distilled_lora")
        if override_path is not None:
            return maybe_download_lora(override_path)

        weight_name = server_args.pipeline_config.stage_2_distilled_lora_weight_name
        lora_path = os.path.join(self.model_path, weight_name)
        if not os.path.exists(lora_path):
            raise ValueError(
                "LTX-2 two-stage pipeline requires the distilled LoRA weights at "
                f"{lora_path}, but the file was not found."
            )
        return lora_path

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, object]:
        modules = super().load_modules(server_args, loaded_modules)

        if loaded_modules is not None and (
            "spatial_upsampler" in loaded_modules
            or "latent_upsampler" in loaded_modules
        ):
            upsampler = loaded_modules.get("spatial_upsampler") or loaded_modules.get(
                "latent_upsampler"
            )
            modules["spatial_upsampler"] = upsampler
            modules["latent_upsampler"] = upsampler
            return modules

        component_model_path = self._resolve_component_path(
            server_args, "spatial_upsampler", "latent_upsampler"
        )
        if not os.path.exists(component_model_path):
            raise ValueError(
                "LTX-2 two-stage pipeline requires a `latent_upsampler` subfolder, "
                "`--spatial-upsampler-path`, or `--latent-upsampler-path`, but none was found."
            )

        latent_upsampler, memory_usage = PipelineComponentLoader.load_component(
            component_name="spatial_upsampler",
            component_model_path=component_model_path,
            transformers_or_diffusers="diffusers",
            server_args=server_args,
        )
        self.memory_usages["spatial_upsampler"] = memory_usage
        self.memory_usages["latent_upsampler"] = memory_usage
        modules["spatial_upsampler"] = latent_upsampler
        modules["latent_upsampler"] = latent_upsampler
        return modules

    @staticmethod
    def _should_merge_stage2_distilled_lora(server_args: ServerArgs) -> bool:
        return is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        super().initialize_pipeline(server_args)
        server_args.component_paths = _resolve_ltx2_two_stage_component_paths(
            self.model_path, server_args.component_paths
        )
        scheduler = self.get_module("scheduler")
        self.modules["scheduler_stage2"] = LTX2FlowMatchScheduler.from_config(
            scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

    def _create_two_stage_pipeline_stages(
        self,
        server_args: ServerArgs,
        *,
        preserve_conditioned_first_frame: bool,
    ):
        stage_2_lora_path = self._resolve_stage_2_lora_path(server_args)
        stage_2_lora_nickname = (
            server_args.pipeline_config.stage_2_distilled_lora_nickname
        )
        self.add_stages(
            [
                InputValidationStage(),
                LTX2TwoStagePreparationStage(),
            ]
        )
        self.add_stage(
            LTX2Stage2LoRADisableStage(
                pipeline=self,
                lora_path=stage_2_lora_path,
                lora_nickname=stage_2_lora_nickname,
            ),
            stage_name="ltx2_stage2_lora_disable_stage",
        )
        self._add_shared_preprocessing_stages()
        self._add_stage_1_denoising()
        self.add_stages(
            [
                LTX2LatentUpsampleStage(
                    latent_upsampler=self.get_module("latent_upsampler"),
                    vae=self.get_module("vae"),
                ),
                LTX2RefinementLatentPreparationStage(
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    preserve_conditioned_first_frame=preserve_conditioned_first_frame,
                ),
            ]
        )
        self.add_stage(
            LTX2Stage2LoRAEnableStage(
                pipeline=self,
                lora_path=stage_2_lora_path,
                lora_nickname=stage_2_lora_nickname,
            ),
            stage_name="ltx2_stage2_lora_enable_stage",
        )
        self.add_stage(
            LTX2RefinementStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler_stage2"),
                distilled_sigmas=server_args.pipeline_config.stage_2_distilled_sigmas,
                vae=self.get_module("vae"),
                audio_vae=self.get_module("audio_vae"),
            ),
            stage_name="l_t_x2_refinement_stage",
        )
        self._add_decoding_stage()

    def create_pipeline_stages(self, server_args: ServerArgs):
        self._create_two_stage_pipeline_stages(
            server_args, preserve_conditioned_first_frame=True
        )


LTX2TwoStagePipeline = build_ltx2_two_stage_pipeline_cls(
    LTX2ImageToVideoTwoStagesPipeline, LTX2PipelineConfig
)


EntryClass = [
    LTX2Pipeline,
    LTX2TwoStagePipeline,
    LTX2ImageToVideoPipeline,
    LTX2ImageToVideoTwoStagesPipeline,
]
