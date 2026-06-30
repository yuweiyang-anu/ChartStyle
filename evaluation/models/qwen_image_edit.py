"""Qwen-Image-Edit inference backend.

This is the open-source base model reported in the paper, driven through the
diffusers ``QwenImageEditPlusPipeline`` with a two-image input:
``image[0]`` = style reference, ``image[1]`` = content image.

Two inference modes share a single pipeline class:
  * default (``fast=False``) — 40 denoising steps, true_cfg_scale 4.0 (paper setting);
  * fast    (``fast=True``)  — 8 steps via the Qwen-Image Lightning LoRA (true_cfg
                               1.0), for quick previews / high-throughput runs.

A per-instance ``asyncio.Lock`` serialises access to a single GPU pipeline; the
optional ``MultiGPUQwenImageEditModel`` shards work across all visible GPUs with
round-robin scheduling so the 300-pair benchmark runs in parallel.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pydantic
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

try:  # optional: only needed for fast (Lightning) mode
    from diffusers import FlowMatchEulerDiscreteScheduler
except Exception:
    FlowMatchEulerDiscreteScheduler = None  # type: ignore

from models import image_utils
from models.base import StyleTransferModel
from prompts.generation_prompts import STYLE_TRANSFER_PROMPT


class QwenImageEditConfig(pydantic.BaseModel):
    """Inference defaults for Qwen-Image-Edit."""

    # Upstream HuggingFace repo id (the two-image "Plus" pipeline ships under this
    # exact name); this is the real model identifier, not a user-facing label.
    MODEL_ID: str = "Qwen/Qwen-Image-Edit-2509"
    NEGATIVE_PROMPT: str = ""
    NUM_IMAGES_PER_PROMPT: int = 1
    GUIDANCE_SCALE: float = 1.0
    SEED: int = 0
    SYSTEM_PROMPT: str = STYLE_TRANSFER_PROMPT

    # Default (40-step) mode — paper setting.
    NUM_INFERENCE_STEPS: int = 40
    TRUE_CFG_SCALE: float = 4.0

    # Fast (8-step) mode — Lightning LoRA distillation.
    FAST_NUM_INFERENCE_STEPS: int = 8
    FAST_TRUE_CFG_SCALE: float = 1.0
    LIGHTNING_REPO: str = "lightx2v/Qwen-Image-Lightning"
    LIGHTNING_WEIGHT_NAME: str = "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"


class QwenImageEditModel(StyleTransferModel):
    """Single-GPU Qwen-Image-Edit style-transfer pipeline.

    Args:
        fast: use the 8-step Lightning LoRA path instead of the 40-step default.
        num_inference_steps: explicit override (takes precedence over ``fast``).
    """

    def __init__(
        self,
        name: str = "qwen_image_edit",
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        fast: bool = False,
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__(name)
        self.config = QwenImageEditConfig()
        if model_id is not None:
            self.config.MODEL_ID = model_id
        self.fast = fast
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._select_dtype()

        # Resolve the effective step count / cfg for this instance.
        if self.fast:
            self.num_inference_steps = num_inference_steps or self.config.FAST_NUM_INFERENCE_STEPS
            self.true_cfg_scale = self.config.FAST_TRUE_CFG_SCALE
        else:
            self.num_inference_steps = num_inference_steps or self.config.NUM_INFERENCE_STEPS
            self.true_cfg_scale = self.config.TRUE_CFG_SCALE

        self.pipeline = self._load_pipeline()
        # Per-instance lock: protects this pipeline object from concurrent use.
        # Distinct GPU instances hold distinct locks, so they still run in parallel.
        self._generation_lock = asyncio.Lock()

    def _select_dtype(self) -> torch.dtype:
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _load_pipeline(self) -> QwenImageEditPlusPipeline:
        try:
            scheduler = self._lightning_scheduler() if self.fast else None
            kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
            if scheduler is not None:
                kwargs["scheduler"] = scheduler
            pipeline = QwenImageEditPlusPipeline.from_pretrained(self.config.MODEL_ID, **kwargs)
            if self.fast:
                self._fuse_lightning_lora(pipeline)
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(self.device)
            return pipeline
        except Exception as exc:
            raise RuntimeError(f"Failed to load Qwen image edit pipeline: {exc}") from exc

    def _lightning_scheduler(self):
        """Flow-match scheduler tuned for the 8-step Lightning distillation."""
        if FlowMatchEulerDiscreteScheduler is None:
            print("⚠️ FlowMatchEulerDiscreteScheduler unavailable; using the default scheduler")
            return None
        config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        try:
            return FlowMatchEulerDiscreteScheduler.from_config(config)
        except Exception as exc:
            print(f"⚠️ Failed to build Lightning scheduler ({exc}); using the default scheduler")
            return None

    def _fuse_lightning_lora(self, pipeline: QwenImageEditPlusPipeline) -> None:
        try:
            pipeline.load_lora_weights(
                self.config.LIGHTNING_REPO,
                weight_name=self.config.LIGHTNING_WEIGHT_NAME,
                adapter_name="lightning",
            )
            pipeline.set_adapters(["lightning"], adapter_weights=[1.0])
            pipeline.fuse_lora(adapter_names=["lightning"], lora_scale=1.0)
            pipeline.unload_lora_weights()
            print("✅ Fused Qwen-Image Lightning 8-step LoRA")
        except Exception as exc:
            print(f"⚠️ Failed to fuse Lightning LoRA ({exc}); falling back to 40-step inference")
            self.fast = False
            self.num_inference_steps = self.config.NUM_INFERENCE_STEPS
            self.true_cfg_scale = self.config.TRUE_CFG_SCALE

    async def generate(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        **kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        async with self._generation_lock:
            content_rgb = image_utils.alpha_over_white(content_input)
            style_rgb = image_utils.alpha_over_white(style_reference)

            start = time.time()
            prompt_text = self.config.SYSTEM_PROMPT.strip()

            settings = {
                "true_cfg_scale": kwargs.get("true_cfg_scale", self.true_cfg_scale),
                "negative_prompt": kwargs.get("negative_prompt", self.config.NEGATIVE_PROMPT),
                "num_inference_steps": kwargs.get("num_inference_steps", self.num_inference_steps),
                "guidance_scale": kwargs.get("guidance_scale", self.config.GUIDANCE_SCALE),
                "num_images_per_prompt": kwargs.get("num_images_per_prompt", self.config.NUM_IMAGES_PER_PROMPT),
            }
            seed = kwargs.get("seed", self.config.SEED)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            generated_image = await asyncio.to_thread(
                self._run_pipeline, prompt_text, content_rgb, style_rgb, generator, settings
            )

            metadata: Dict[str, Any] = {
                "model_name": self.name,
                "processing_type": "qwen_image_edit_fast" if self.fast else "qwen_image_edit",
                "prompt": prompt_text,
                "seed": seed,
                "fast": self.fast,
                "generation_time": time.time() - start,
                **settings,
            }
            return generated_image, metadata

    def _run_pipeline(
        self,
        prompt_text: str,
        content_image: Image.Image,
        style_image: Image.Image,
        generator: torch.Generator,
        settings: Dict[str, Any],
    ) -> Image.Image:
        inputs = {
            "image": [style_image, content_image],
            "prompt": prompt_text,
            "generator": generator,
            **settings,
        }
        with torch.inference_mode():
            output = self.pipeline(**inputs)
        if not output.images:
            raise RuntimeError("Qwen pipeline returned no images")
        return output.images[0].convert("RGB")


class MultiGPUQwenImageEditModel(StyleTransferModel):
    """Round-robin wrapper that spreads generations across every visible GPU."""

    def __init__(
        self,
        name: str = "qwen_image_edit_multigpu",
        model_id: Optional[str] = None,
        max_gpus: Optional[int] = None,
        fast: bool = False,
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__(name)
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
            if max_gpus is not None:
                gpu_ids = gpu_ids[:max_gpus]
        else:
            gpu_ids = []

        common = dict(model_id=model_id, fast=fast, num_inference_steps=num_inference_steps)
        if not gpu_ids:
            print("⚠️ No CUDA GPUs available; falling back to a single CPU pipeline")
            self.models: List[QwenImageEditModel] = [
                QwenImageEditModel(name=f"{name}_cpu", device="cpu", **common)
            ]
        else:
            print(f"🚀 Loading Qwen pipelines on GPUs: {gpu_ids} (fast={fast})")
            self.models = [
                QwenImageEditModel(name=f"{name}_cuda{i}", device=f"cuda:{i}", **common)
                for i in gpu_ids
            ]

        self._index = 0
        self._lock = threading.Lock()

    def _next_model(self) -> QwenImageEditModel:
        with self._lock:
            model = self.models[self._index]
            self._index = (self._index + 1) % len(self.models)
            return model

    async def generate(
        self,
        style_reference: Image.Image,
        content_input: Image.Image,
        **kwargs: Any,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        return await self._next_model().generate(style_reference, content_input, **kwargs)
