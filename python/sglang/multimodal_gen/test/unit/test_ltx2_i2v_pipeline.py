import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2ImageToVideoTwoStagesPipelineConfig,
    calculate_ltx2_mu,
    get_ltx2_packed_video_seq_len,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2Attention,
    LTX2VideoTransformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_two_stage import (
    LTX2LatentUpsampleStage,
    LTX2RefinementLatentPreparationStage,
    LTX2Stage2LoRAControlStage,
    LTX2TwoStagePreparationStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


def _make_batch(**kwargs) -> Req:
    sampling_params = LTX2SamplingParams(
        prompt="test",
        seed=42,
        num_outputs_per_prompt=1,
        height=kwargs.pop("height", 512),
        width=kwargs.pop("width", 896),
        num_frames=kwargs.pop("num_frames", 121),
        fps=kwargs.pop("fps", 25),
        num_inference_steps=kwargs.pop("num_inference_steps", 40),
        guidance_scale=kwargs.pop("guidance_scale", 4.0),
        generate_audio=kwargs.pop("generate_audio", True),
    )
    batch = Req(sampling_params=sampling_params, **kwargs)
    if batch.prompt_embeds == []:
        batch.prompt_embeds = torch.zeros((1, 4), dtype=torch.float32)
    if batch.image_embeds == []:
        batch.image_embeds = []
    return batch


class _FakeScheduler:
    def __init__(self):
        self.sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.timesteps = torch.tensor([999.0, 0.0], dtype=torch.float32)
        self.num_inference_steps = 2
        self.last_kwargs = None

    def set_timesteps(self, sigmas, device, **kwargs):
        self.last_kwargs = {"sigmas": list(sigmas), "device": device, **kwargs}
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        self.timesteps = torch.arange(
            len(sigmas), 0, -1, dtype=torch.float32, device=device
        )
        self.num_inference_steps = len(sigmas)


class _FakeVAE:
    def __init__(self, channels: int):
        self.latents_mean = torch.zeros(channels, dtype=torch.float32)
        self.latents_std = torch.ones(channels, dtype=torch.float32)
        self.config = SimpleNamespace(scaling_factor=1.0)


class _FakeAudioVAE:
    def __init__(self, features: int):
        self.latents_mean = torch.zeros(features, dtype=torch.float32)
        self.latents_std = torch.ones(features, dtype=torch.float32)


class _IdentityUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls = 0

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(q)


class _FakeLinear(nn.Module):
    def __init__(self, _input_size, output_size, **_kwargs):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        if x.shape[-1] == self.output_size:
            return x, None
        if x.shape[-1] > self.output_size:
            return x[..., : self.output_size], None
        padded = torch.nn.functional.pad(x, (0, self.output_size - x.shape[-1]))
        return padded, None


class _CountingFakeLinear(_FakeLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        return super().forward(x)


class _CountingProjection(nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x + self.bias


class _FakeLoRAPipeline:
    def __init__(self):
        self.merged = False
        self.set_lora_calls = 0
        self.merge_calls = 0
        self.unmerge_calls = 0

    def set_lora(self, *args, **kwargs):
        self.set_lora_calls += 1
        self.merged = True

    def merge_lora_weights(self, target="all", strength=1.0):
        self.merge_calls += 1
        self.merged = True

    def unmerge_lora_weights(self, target="all"):
        self.unmerge_calls += 1
        self.merged = False

    def is_lora_effective(self, target="all") -> bool:
        return self.merged


class _FakeJointModel:
    def __init__(self):
        self.calls = 0
        self.last_hidden_batch = None
        self.last_mask_batch = None

    def __call__(
        self,
        *,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        encoder_attention_mask=None,
        **kwargs,
    ):
        self.calls += 1
        self.last_hidden_batch = int(hidden_states.shape[0])
        self.last_mask_batch = (
            None
            if encoder_attention_mask is None
            else int(encoder_attention_mask.shape[0])
        )
        video_bias = encoder_hidden_states[:, :1, :1].to(hidden_states.dtype)
        audio_bias = audio_encoder_hidden_states[:, :1, :1].to(
            audio_hidden_states.dtype
        )
        return (
            hidden_states + video_bias.expand_as(hidden_states),
            audio_hidden_states + audio_bias.expand_as(audio_hidden_states),
        )


class TestLTX2MuCalculation(unittest.TestCase):
    def setUp(self):
        self.pipeline_config = LTX2ImageToVideoTwoStagesPipelineConfig()

    def test_one_stage_seq_len_and_mu(self):
        batch = _make_batch(width=896, height=512, num_frames=121)

        seq_len = get_ltx2_packed_video_seq_len(batch, self.pipeline_config)
        self.assertEqual(seq_len, 222208)
        self.assertAlmostEqual(
            calculate_ltx2_mu(batch, self.pipeline_config),
            80.14999999999999,
        )

    def test_two_stage_stage1_seq_len_and_mu(self):
        batch = _make_batch(width=1920, height=1088, num_frames=121)
        batch.height //= 2
        batch.width //= 2

        seq_len = get_ltx2_packed_video_seq_len(batch, self.pipeline_config)
        self.assertEqual(seq_len, 252960)
        self.assertAlmostEqual(
            calculate_ltx2_mu(batch, self.pipeline_config),
            91.16145833333333,
        )

    def test_two_stage_stage2_seq_len_and_mu(self):
        batch = _make_batch(width=1920, height=1088, num_frames=121)

        seq_len = get_ltx2_packed_video_seq_len(batch, self.pipeline_config)
        self.assertEqual(seq_len, 1011840)
        self.assertAlmostEqual(
            calculate_ltx2_mu(batch, self.pipeline_config),
            362.8958333333333,
        )


class TestLTX2RefinementStage(unittest.TestCase):
    def setUp(self):
        self.global_args = MagicMock(
            comfyui_mode=False,
            enable_torch_compile=False,
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(hidden_size=8, num_attention_heads=2)
            ),
        )
        self.server_args = MagicMock(
            pipeline_config=LTX2ImageToVideoTwoStagesPipelineConfig()
        )

    def test_refinement_stage_recomputes_mu_and_restores_state(self):
        scheduler = _FakeScheduler()
        batch = _make_batch(width=1920, height=1088, num_frames=121)
        batch.latents = torch.zeros((1, 8, 16), dtype=torch.float32)
        batch.timesteps = torch.tensor([10.0, 5.0], dtype=torch.float32)
        batch.sigmas = [0.6, 0.2]
        batch.num_inference_steps = 2
        batch.extra["mu"] = -1.0

        captured = {}

        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_attn_backend",
            return_value=MagicMock(),
        ):
            stage = LTX2RefinementStage(
                transformer=MagicMock(),
                scheduler=scheduler,
                distilled_sigmas=(0.9, 0.4, 0.0),
            )

        def _fake_forward(_self, batch_arg, _server_args):
            captured["timesteps"] = batch_arg.timesteps.clone()
            captured["sigmas"] = list(batch_arg.sigmas)
            captured["num_inference_steps"] = batch_arg.num_inference_steps
            return batch_arg

        with patch.object(
            LTX2AVDenoisingStage,
            "forward",
            autospec=True,
            side_effect=_fake_forward,
        ):
            result = stage.forward(batch, self.server_args)

        self.assertIs(result, batch)
        self.assertAlmostEqual(batch.extra["mu"], 362.8958333333333)
        self.assertAlmostEqual(scheduler.last_kwargs["mu"], 362.8958333333333)
        self.assertEqual(len(captured["sigmas"]), 3)
        self.assertAlmostEqual(captured["sigmas"][0], 0.9)
        self.assertAlmostEqual(captured["sigmas"][1], 0.4)
        self.assertAlmostEqual(captured["sigmas"][2], 0.0)
        self.assertEqual(captured["num_inference_steps"], 3)
        self.assertTrue(torch.equal(batch.timesteps, torch.tensor([10.0, 5.0])))
        self.assertEqual(batch.sigmas, [0.6, 0.2])
        self.assertEqual(batch.num_inference_steps, 2)
        self.assertTrue(torch.equal(scheduler.timesteps, torch.tensor([999.0, 0.0])))
        self.assertTrue(
            torch.equal(scheduler.sigmas, torch.tensor([1.0, 0.0], dtype=torch.float32))
        )
        self.assertEqual(scheduler.num_inference_steps, 2)


class TestLTX2AVDenoisingCfgBatching(unittest.TestCase):
    def setUp(self):
        self.global_args = MagicMock(
            comfyui_mode=False,
            enable_torch_compile=False,
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(hidden_size=8, num_attention_heads=2)
            ),
        )

    def test_cfg_branch_uses_single_batched_model_call(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_attn_backend",
            return_value=MagicMock(),
        ):
            stage = LTX2AVDenoisingStage(
                transformer=MagicMock(),
                scheduler=MagicMock(),
            )

        model = _FakeJointModel()
        latent_model_input = torch.zeros((1, 3, 2), dtype=torch.float32)
        audio_latent_model_input = torch.zeros((1, 2, 4), dtype=torch.float32)
        encoder_hidden_states = torch.ones((1, 5, 6), dtype=torch.float32)
        negative_encoder_hidden_states = torch.full((1, 5, 6), 2.0, dtype=torch.float32)
        audio_encoder_hidden_states = torch.full((1, 5, 6), 3.0, dtype=torch.float32)
        negative_audio_encoder_hidden_states = torch.full(
            (1, 5, 6), 4.0, dtype=torch.float32
        )
        encoder_attention_mask = torch.ones((1, 5), dtype=torch.int64)
        negative_encoder_attention_mask = torch.zeros((1, 5), dtype=torch.int64)

        v_pos, a_pos, v_neg, a_neg = stage._forward_joint_model_with_optional_cfg(
            current_model=model,
            latent_model_input=latent_model_input,
            audio_latent_model_input=audio_latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep_video=torch.tensor([1.0], dtype=torch.float32),
            timestep_audio=torch.tensor([1.0], dtype=torch.float32),
            latent_num_frames=16,
            latent_height=18,
            latent_width=32,
            fps=25,
            audio_num_frames_latent=121,
            video_coords=None,
            audio_coords=None,
            do_classifier_free_guidance=True,
            negative_encoder_hidden_states=negative_encoder_hidden_states,
            negative_audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
            negative_encoder_attention_mask=negative_encoder_attention_mask,
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(model.last_hidden_batch, 2)
        self.assertEqual(model.last_mask_batch, 2)
        self.assertTrue(torch.allclose(v_pos, torch.ones_like(v_pos)))
        self.assertTrue(torch.allclose(v_neg, torch.full_like(v_neg, 2.0)))
        self.assertTrue(torch.allclose(a_pos, torch.full_like(a_pos, 3.0)))
        self.assertTrue(torch.allclose(a_neg, torch.full_like(a_neg, 4.0)))

    def test_prepare_cfg_conditioning_hoists_static_prompt_tensors(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_attn_backend",
            return_value=MagicMock(),
        ):
            stage = LTX2AVDenoisingStage(
                transformer=MagicMock(),
                scheduler=MagicMock(),
            )

        encoder_hidden_states = torch.ones((1, 5, 6), dtype=torch.float32)
        negative_encoder_hidden_states = torch.full((1, 5, 6), 2.0, dtype=torch.float32)
        audio_encoder_hidden_states = torch.full((1, 5, 6), 3.0, dtype=torch.float32)
        negative_audio_encoder_hidden_states = torch.full(
            (1, 5, 6), 4.0, dtype=torch.float32
        )
        encoder_attention_mask = torch.ones((1, 5), dtype=torch.int64)
        negative_encoder_attention_mask = torch.zeros((1, 5), dtype=torch.int64)

        (
            cfg_encoder_hidden_states,
            cfg_audio_encoder_hidden_states,
            cfg_attention_mask,
        ) = stage._prepare_cfg_conditioning(
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            do_classifier_free_guidance=True,
            negative_encoder_hidden_states=negative_encoder_hidden_states,
            negative_audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
            negative_encoder_attention_mask=negative_encoder_attention_mask,
        )

        self.assertEqual(cfg_encoder_hidden_states.shape[0], 2)
        self.assertEqual(cfg_audio_encoder_hidden_states.shape[0], 2)
        self.assertEqual(cfg_attention_mask.shape[0], 2)
        self.assertTrue(
            torch.equal(cfg_encoder_hidden_states[0], encoder_hidden_states[0])
        )
        self.assertTrue(
            torch.equal(cfg_encoder_hidden_states[1], negative_encoder_hidden_states[0])
        )

    def test_combine_cfg_velocity_matches_guided_velocity_formula(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_attn_backend",
            return_value=MagicMock(),
        ):
            stage = LTX2AVDenoisingStage(
                transformer=MagicMock(),
                scheduler=MagicMock(),
            )

        positive = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        negative = torch.tensor([[0.25, 1.25]], dtype=torch.float16)
        guidance_delta = 3.0

        combined = stage._combine_cfg_velocity(positive, negative, guidance_delta)
        expected = positive.float() + guidance_delta * (
            positive.float() - negative.float()
        )

        self.assertEqual(combined.dtype, torch.float32)
        self.assertTrue(torch.allclose(combined, expected))


class TestLTX2TwoStageFlow(unittest.TestCase):
    def setUp(self):
        self.global_args = MagicMock(comfyui_mode=False)
        self.server_args = MagicMock(
            pipeline_config=LTX2ImageToVideoTwoStagesPipelineConfig(),
            lora_path=None,
        )

    def test_stage_dimensions_restore_and_cfg_is_disabled(self):
        batch = _make_batch(width=1920, height=1088, num_frames=121)
        batch.latents = torch.ones((1, 2, 16, 17, 30), dtype=torch.float32)
        batch.audio_latents = torch.ones((1, 121, 16), dtype=torch.float32)
        batch.image_latent = torch.ones((1, 2, 4), dtype=torch.float32)
        batch.condition_image = torch.ones((1, 3, 8, 8), dtype=torch.float32)
        batch.ltx2_num_image_tokens = 5
        batch.generator = torch.Generator(device="cpu").manual_seed(0)

        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args):
            prep_stage = LTX2TwoStagePreparationStage()
            upsample_stage = LTX2LatentUpsampleStage(
                latent_upsampler=_IdentityUpsampler(),
                vae=_FakeVAE(channels=2),
            )
            refinement_prep_stage = LTX2RefinementLatentPreparationStage(
                vae=_FakeVAE(channels=2),
                audio_vae=_FakeAudioVAE(features=16),
            )

        batch = prep_stage.forward(batch, self.server_args)
        self.assertEqual((batch.height, batch.width), (544, 960))

        batch = upsample_stage.forward(batch, self.server_args)
        self.assertEqual((batch.height, batch.width), (1088, 1920))

        batch = refinement_prep_stage.forward(batch, self.server_args)
        self.assertEqual(batch.raw_latent_shape, batch.latents.shape)
        self.assertEqual(batch.raw_audio_latent_shape, batch.audio_latents.shape)
        self.assertFalse(batch.do_classifier_free_guidance)
        self.assertEqual(batch.guidance_scale, 1.0)
        self.assertIsNone(batch.image_latent)
        self.assertIsNone(batch.condition_image)
        self.assertEqual(batch.ltx2_num_image_tokens, 0)
        self.assertEqual(batch.num_inference_steps, 4)

    def test_missing_stage2_lora_path_fails_early(self):
        pipeline = _FakeLoRAPipeline()
        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args):
            stage = LTX2Stage2LoRAControlStage(
                pipeline=pipeline,
                enable=False,
                lora_path="/tmp/definitely-missing-stage2-lora.safetensors",
            )

        batch = _make_batch()
        with self.assertRaisesRegex(ValueError, "distilled LoRA weights"):
            stage.forward(batch, self.server_args)


class TestLTX2Stage2LoRAControl(unittest.TestCase):
    def setUp(self):
        self.global_args = MagicMock(comfyui_mode=False)
        self.lora_path = "/tmp/ltx2-stage2-lora.safetensors"
        self.server_args = MagicMock(lora_path=None)

    def test_internal_lora_is_preloaded_once_then_toggled(self):
        pipeline = _FakeLoRAPipeline()
        batch = _make_batch()

        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args):
            disable_stage = LTX2Stage2LoRAControlStage(
                pipeline=pipeline,
                enable=False,
                lora_path=self.lora_path,
                lora_nickname="stage_2_distilled",
            )
            enable_stage = LTX2Stage2LoRAControlStage(
                pipeline=pipeline,
                enable=True,
                lora_path=self.lora_path,
                lora_nickname="stage_2_distilled",
            )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_two_stage.os.path.exists",
            return_value=True,
        ):
            disable_stage.forward(batch, self.server_args)
            self.assertEqual(pipeline.set_lora_calls, 1)
            self.assertEqual(pipeline.unmerge_calls, 1)
            self.assertFalse(pipeline.merged)

            enable_stage.forward(batch, self.server_args)
            self.assertEqual(pipeline.set_lora_calls, 1)
            self.assertEqual(pipeline.merge_calls, 1)
            self.assertTrue(pipeline.merged)

            disable_stage.forward(batch, self.server_args)
            self.assertEqual(pipeline.set_lora_calls, 1)
            self.assertEqual(pipeline.unmerge_calls, 2)
            self.assertFalse(pipeline.merged)

            enable_stage.forward(batch, self.server_args)
            self.assertEqual(pipeline.set_lora_calls, 1)
            self.assertEqual(pipeline.merge_calls, 2)

    def test_external_lora_bypasses_internal_stage2_switching(self):
        pipeline = _FakeLoRAPipeline()
        server_args = MagicMock(lora_path="/tmp/external-lora.safetensors")

        with patch(_GLOBAL_ARGS_PATCH, return_value=self.global_args):
            stage = LTX2Stage2LoRAControlStage(
                pipeline=pipeline,
                enable=True,
                lora_path=self.lora_path,
            )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_two_stage.os.path.exists",
            return_value=True,
        ):
            stage.forward(_make_batch(), server_args)

        self.assertEqual(pipeline.set_lora_calls, 0)
        self.assertEqual(pipeline.merge_calls, 0)
        self.assertEqual(pipeline.unmerge_calls, 0)


class TestLTX2Attention(unittest.TestCase):
    def test_unmasked_path_uses_attention_backend(self):
        with patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.ColumnParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.RowParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.USPAttention",
            _FakeAttention,
        ):
            layer = LTX2Attention(query_dim=4, heads=2, dim_head=2)
        fake_attn = _FakeAttention()
        layer.attn = fake_attn

        x = torch.randn(1, 3, 4)
        _ = layer(x)

        self.assertEqual(fake_attn.calls, 1)

    def test_masked_path_uses_sdpa(self):
        with patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.ColumnParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.RowParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.USPAttention",
            _FakeAttention,
        ):
            layer = LTX2Attention(query_dim=4, heads=2, dim_head=2)
        fake_attn = _FakeAttention()
        layer.attn = fake_attn
        x = torch.randn(1, 3, 4)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)

        with patch(
            "torch.nn.functional.scaled_dot_product_attention",
            autospec=True,
            return_value=torch.zeros((1, 2, 3, 2), dtype=x.dtype),
        ) as sdpa_mock:
            _ = layer(x, mask=mask)

        self.assertEqual(fake_attn.calls, 0)
        self.assertEqual(sdpa_mock.call_count, 1)
        attn_mask = sdpa_mock.call_args.kwargs["attn_mask"]
        self.assertEqual(attn_mask.shape, (1, 1, 1, 3))

    def test_prompt_cross_attention_cache_reuses_projected_context_kv(self):
        with patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.ColumnParallelLinear",
            _CountingFakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.RowParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.USPAttention",
            _FakeAttention,
        ):
            layer = LTX2Attention(query_dim=4, context_dim=4, heads=2, dim_head=2)

        x = torch.randn(1, 3, 4)
        context = torch.randn(1, 5, 4)
        _ = layer(x, context=context, cache_context_kv=True)
        _ = layer(x, context=context, cache_context_kv=True)

        self.assertEqual(layer.to_k.calls, 1)
        self.assertEqual(layer.to_v.calls, 1)

    def test_reset_runtime_attention_cache_invalidates_prompt_kv_cache(self):
        with patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.ColumnParallelLinear",
            _CountingFakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.RowParallelLinear",
            _FakeLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.USPAttention",
            _FakeAttention,
        ):
            layer = LTX2Attention(query_dim=4, context_dim=4, heads=2, dim_head=2)

        x = torch.randn(1, 3, 4)
        context = torch.randn(1, 5, 4)
        _ = layer(x, context=context, cache_context_kv=True)
        layer.reset_runtime_attention_cache()
        _ = layer(x, context=context, cache_context_kv=True)

        self.assertEqual(layer.to_k.calls, 2)
        self.assertEqual(layer.to_v.calls, 2)

    def test_prompt_projection_cache_reuses_projected_prompt_embeddings(self):
        fake_model = SimpleNamespace(
            _runtime_prompt_projection_cache=None,
            transformer_blocks=[],
            caption_projection=_CountingProjection(1.0),
            audio_caption_projection=_CountingProjection(2.0),
        )
        encoder_hidden_states = torch.randn(1, 5, 4)
        audio_encoder_hidden_states = torch.randn(1, 5, 4)

        projected_video_1, projected_audio_1 = (
            LTX2VideoTransformer3DModel._project_prompt_embeddings_with_cache(
                fake_model, encoder_hidden_states, audio_encoder_hidden_states
            )
        )
        projected_video_2, projected_audio_2 = (
            LTX2VideoTransformer3DModel._project_prompt_embeddings_with_cache(
                fake_model, encoder_hidden_states, audio_encoder_hidden_states
            )
        )

        self.assertEqual(fake_model.caption_projection.calls, 1)
        self.assertEqual(fake_model.audio_caption_projection.calls, 1)
        self.assertTrue(torch.equal(projected_video_1, projected_video_2))
        self.assertTrue(torch.equal(projected_audio_1, projected_audio_2))

        LTX2VideoTransformer3DModel.reset_runtime_attention_cache(fake_model)
        _ = LTX2VideoTransformer3DModel._project_prompt_embeddings_with_cache(
            fake_model, encoder_hidden_states, audio_encoder_hidden_states
        )

        self.assertEqual(fake_model.caption_projection.calls, 2)
        self.assertEqual(fake_model.audio_caption_projection.calls, 2)


if __name__ == "__main__":
    unittest.main()
