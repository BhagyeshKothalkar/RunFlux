import base64
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import modal
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for Model, Hardware, and Paths."""

    model_name: str = "flux.2-klein-4b"
    safe_model_name: str = "flux_2_klein_4b"
    gpu: str = "L40S"
    compilation_suffix: str = "O3"
    ckpts_path: str = "/checkpoints"
    aot_path: str = "/artifacts"

    @property
    def package_path(self) -> str:
        return os.path.join(
            self.aot_path,
            self.safe_model_name,
            f"{self.safe_model_name}_{self.gpu}_{self.compilation_suffix}.pt2",
        )


@dataclass
class InferenceConfig:
    """Configuration for Inference / Generation."""

    prompt: str = "A high-quality image"
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 4
    guidance: float = 2.8
    batch_size: int = 1  # Unified batch size for compiler and inference

    def copy(self) -> "InferenceConfig":
        return InferenceConfig(
            prompt=self.prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            num_steps=self.num_steps,
            guidance=self.guidance,
            batch_size=self.batch_size,
        )


# Initialize configs
PIPELINE_CFG = PipelineConfig()
INFERENCE_DEFAULTS = InferenceConfig()

# ----------------------------------------------------------------------------
# Modal Infrastructure
# ----------------------------------------------------------------------------

ckpts_vol = modal.Volume.from_name("flux2_ckpts", create_if_missing=True)
inductor_vol = modal.Volume.from_name("inductor_aot_models", create_if_missing=True)
inductor_cache_vol = modal.Volume.from_name("inductor-cache", create_if_missing=True)
nv_cache_vol = modal.Volume.from_name("nv-cache", create_if_missing=True)
triton_cache_vol = modal.Volume.from_name("triton-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel")
    .apt_install("git", "curl")
    .uv_pip_install(
        "git+https://github.com/BhagyeshKothalkar/flux2",
    )
    .env(
        {
            "HF_HUB_CACHE": PIPELINE_CFG.ckpts_path,
            "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
            "TRITON_CACHE_DIR": "/root/.triton",
            "CUDA_CACHE_PATH": "/root/.nv",
        }
    )
    .uv_pip_install("kernels<0.13.0", "torchao")
)

app = modal.App("flux2-klein-pipeline")

# ----------------------------------------------------------------------------
# Utilities & Model Wrappers
# ----------------------------------------------------------------------------


class ModelWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        return self.transformer(
            x,
            x_ids.contiguous().to(torch.bfloat16),
            timesteps.contiguous().to(torch.bfloat16),
            ctx.contiguous().to(torch.bfloat16),
            ctx_ids.contiguous().to(torch.bfloat16),
            guidance.contiguous().to(torch.bfloat16),
        )


def img_to_b64_string(image):
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    im_data = buffer.getvalue()
    im_b64 = base64.b64encode(im_data)
    return im_b64.decode(encoding="utf-8")


def save_image_bytes(img_bytes, save_name="output.png"):
    with open(save_name, "wb") as f:
        f.write(img_bytes)
    print(f"Saved image to {save_name}")


# ----------------------------------------------------------------------------
# Inference Endpoint
# ----------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=PIPELINE_CFG.gpu,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        PIPELINE_CFG.ckpts_path: ckpts_vol,
        PIPELINE_CFG.aot_path: inductor_vol,
        "/root/.nv": nv_cache_vol,
        "/root/.triton": triton_cache_vol,
        "/root/.inductor-cache": inductor_cache_vol,
    },
)
class FluxRun:
    @modal.enter()
    def enter(self):
        from flux2.util import FLUX2_MODEL_INFO, load_ae, load_text_encoder

        self.model_info = FLUX2_MODEL_INFO[PIPELINE_CFG.model_name]
        self.device = torch.device("cuda")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")

        self.text_encoder = load_text_encoder(PIPELINE_CFG.model_name, self.device)
        self.package_path = PIPELINE_CFG.package_path

        self.loaded_model = ModelWrapper(
            torch._inductor.aoti_load_package(self.package_path)
        )
        self.ae = load_ae(PIPELINE_CFG.model_name)

        self.loaded_model.eval()
        self.ae.eval()
        self.text_encoder.eval()

        # Load Inference Configs
        self.cfg = INFERENCE_DEFAULTS.copy()
        defaults = self.model_info.get("defaults", {})
        if "num_steps" in defaults:
            self.cfg.num_steps = defaults["num_steps"]
        if "guidance" in defaults:
            self.cfg.guidance = defaults["guidance"]

    @modal.method()
    def infer(self, prompt: str, cond_image_b64: str):
        import random
        import tempfile

        from einops import rearrange
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            encode_image_refs,
            get_schedule,
            scatter_ids,
        )
        from PIL import Image

        t0 = time.perf_counter()
        if not prompt or prompt.strip() == "":
            prompt = self.cfg.prompt
        else:
            self.cfg.prompt = prompt

        cond_image_bytes = base64.b64decode(cond_image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(cond_image_bytes)
            tmp.flush()

            img = Image.open(tmp.name)

            # Use unified batch size
            batch_size = self.cfg.batch_size
            img_ctx: List[Image.Image] = [img] * batch_size
            prompt_batch = [prompt] * batch_size

            seed = (
                self.cfg.seed if self.cfg.seed is not None else random.randrange(2**31)
            )

            t1 = time.perf_counter()
            print(f"Configuration took {t1 - t0:.4f}s")

            with torch.no_grad():
                ref_tokens, ref_ids = encode_image_refs(self.ae, img_ctx)

                if ref_tokens is not None and ref_ids is not None:
                    if ref_tokens.shape[0] != batch_size:
                        ref_tokens = ref_tokens.expand(batch_size, -1, -1).contiguous()
                        ref_ids = ref_ids.expand(batch_size, -1, -1).contiguous()

                ctx = self.text_encoder(prompt_batch).to(torch.bfloat16)
                ctx, ctx_ids = batched_prc_txt(ctx)

                shape = (batch_size, 128, self.cfg.height // 16, self.cfg.width // 16)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(
                    shape, generator=generator, dtype=torch.bfloat16, device="cuda"
                )

                x, x_ids = batched_prc_img(randn)
                timesteps = get_schedule(self.cfg.num_steps, x.shape[1])

                t2 = time.perf_counter()
                print(f"Pre-denoising processing took {t2 - t1:.4f}s")

                x = denoise(
                    self.loaded_model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=self.cfg.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )

                t3 = time.perf_counter()
                print(f"Denoising took {t3 - t2:.4f}s")

                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = self.ae.decode(x).float()
                x = x.clamp(-1, 1)
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                image_b64 = img_to_b64_string(img)

                t4 = time.perf_counter()
                print(f"Total generation took {t4 - t0:.4f}s")

                return image_b64


# ----------------------------------------------------------------------------
# Compiler Endpoint
# ----------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=PIPELINE_CFG.gpu,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        PIPELINE_CFG.ckpts_path: ckpts_vol,
        PIPELINE_CFG.aot_path: inductor_vol,
        "/root/.nv": nv_cache_vol,
        "/root/.triton": triton_cache_vol,
        "/root/.inductor-cache": inductor_cache_vol,
    },
    timeout=3000,
)
class Compiler:
    @modal.enter()
    def enter(self):
        import multiprocessing

        import torch._inductor.config as inductor_config
        from flux2.util import load_flow_model

        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")

        self.model = load_flow_model(PIPELINE_CFG.model_name, device=self.device)
        ckpts_vol.commit()
        self.model.eval()

        # Use config batch size
        self.b = INFERENCE_DEFAULTS.batch_size

        # Dummy inputs matching the batch size
        self.x = torch.rand(
            (self.b, 4096, 128), device=self.device, dtype=torch.bfloat16
        )
        self.x_ids = torch.rand(
            (self.b, 4096, 4), device=self.device, dtype=torch.bfloat16
        )
        self.ctx = torch.rand(
            (self.b, 512, 7680), device=self.device, dtype=torch.bfloat16
        )
        self.ctx_ids = torch.rand(
            (self.b, 512, 4), device=self.device, dtype=torch.bfloat16
        )
        self.timesteps = torch.rand((self.b,), device=self.device, dtype=torch.bfloat16)
        self.guidance = torch.full(
            (self.b,), 1.0, device=self.device, dtype=torch.bfloat16
        )
        self.ref_tokens = torch.rand(
            (self.b, 4096, 128), device=self.device, dtype=torch.bfloat16
        )
        self.ref_ids = torch.rand(
            (self.b, 4096, 4), device=self.device, dtype=torch.bfloat16
        )

        self.x = torch.cat((self.x, self.ref_tokens), dim=1)
        self.x_ids = torch.cat((self.x_ids, self.ref_ids), dim=1)

        self.dummy_args = (
            self.x,
            self.x_ids,
            self.timesteps,
            self.ctx,
            self.ctx_ids,
            self.guidance,
        )

        self.package_path = PIPELINE_CFG.package_path
        os.makedirs(os.path.dirname(self.package_path), exist_ok=True)

        # Inductor compilation settings
        inductor_config.compile_threads = multiprocessing.cpu_count()
        inductor_config.fx_graph_cache = True
        inductor_config.autotune_local_cache = True
        inductor_config.disable_progress = False
        inductor_config.max_autotune = True
        inductor_config.freezing = True
        inductor_config.coordinate_descent_tuning = True
        inductor_config.layout_optimization = True
        inductor_config.triton.cudagraphs = True
        inductor_config.triton.cudagraph_trees = False

        inductor_config.aot_inductor.compile_wrapper_opt_level = (
            PIPELINE_CFG.compilation_suffix
        )
        inductor_config.cuda.enable_cuda_lto = True
        inductor_config.aot_inductor.emit_multi_arch_kernel = False
        inductor_config.coordinate_descent_check_all_directions = True
        inductor_config.epilogue_fusion = True
        inductor_config.triton.multi_kernel = 0
        inductor_config.triton.store_cubin = True
        inductor_config.aot_inductor.package = True

        os.environ["TORCH_INDUCTOR_CPP_VEC_ISA"] = "avx2"
        inductor_config.cpp.vec_isa_ok = False

    @modal.method()
    def compile(self):
        print("Starting compilation")

        from torch.export import export
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
            """
            Filters modules to quantize based on the exact layer targets
            found in the provided quantization metadata.
            """
            # Only target linear layers
            if not isinstance(module, torch.nn.Linear):
                return False

            parts = fqn.split(".")

            # Match double_blocks targets
            # e.g., double_blocks.0.img_attn.proj, double_blocks.1.txt_mlp.2
            if len(parts) == 4 and parts[0] == "double_blocks":
                _, _, block_type, layer_name = parts

                if block_type in ("img_attn", "txt_attn"):
                    return layer_name in ("proj", "qkv")

                elif block_type in ("img_mlp", "txt_mlp"):
                    return layer_name in ("0", "2")

            # Match single_blocks targets
            # e.g., single_blocks.0.linear1, single_blocks.19.linear2
            elif len(parts) == 3 and parts[0] == "single_blocks":
                _, _, layer_name = parts
                return layer_name in ("linear1", "linear2")

            return False

        with torch.no_grad():
            try:
                print("Quantizing...")
                quantize_(
                    self.model,
                    Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                    filter_fn=filter_fn,
                )

                print("Exporting...")
                exported_program = export(self.model, self.dummy_args, strict=False)

                print("AOT Compiling to .pt2 (Fast Mode)...")
                output_path = torch._inductor.aoti_compile_and_package(
                    exported_program,
                    package_path=self.package_path,
                )

                print("Committing volumes...")
                inductor_vol.commit()
                nv_cache_vol.commit()
                triton_cache_vol.commit()
                inductor_cache_vol.commit()

                print(f"Compilation finished. Saved to: {output_path}")
                return output_path

            except Exception as e:
                print(f"Error while compiling: {e}")
                raise e


# ----------------------------------------------------------------------------
# Local Entrypoint
# ----------------------------------------------------------------------------

edit_prompt = """
generate the image of the same person in the same setting with the following changes: 
the foot should be planted horizontally on the ground. Currently, the feet appear raised 
as seen from the raised angle of the  shoes and the reflection of the shoes that is cast at a distance
and at a different angle from the shoes. Change this to firmly plant both heels horizontally on the ground.
Thus, both shoes must appear to be 
horizontally placed on the ground and change the reflection of the footwear in floor to 
be directly vertically below it in the same orientation as the shoes. Slightly adjust the legs and hip placement of the person
to biomechanically natural with the heels touching the ground. keep the person, facial expressions, physique of the person, and the 
overall appearance of the exercise. Make sure that the person's identity, equipment, background are exactly preserved
 preserve rest of the details as exactly in the original image. 
"""


@app.local_entrypoint()
def main():
    from PIL import Image

    input_image_path = "assets/input/input.png"
    prompt = edit_prompt

    input_image = Image.open(input_image_path).resize(
        (INFERENCE_DEFAULTS.width, INFERENCE_DEFAULTS.height)
    )

    instance = FluxRun()
    # compiler_instance = Compiler()
    # handle = compiler_instance.compile.spawn()

    # try:
    #     while True:
    #         try:
    #             handle.get(timeout=30)
    #             print("Compilation complete.")
    #             break
    #         except TimeoutError:
    #             print(
    #                 f"--- [Local Heartbeat] Still waiting for {handle.object_id}... ---"
    #             )
    #             continue
    # except Exception as e:
    #     print(f"Task failed or timed out: {e}")

    # print("Sleeping to ensure sync...")
    # time.sleep(10)

    b64_string = img_to_b64_string(input_image)
    output_b64_string = instance.infer.remote(prompt, b64_string)
    output_bytes = base64.b64decode(output_b64_string)

    # Ensure output directory exists before saving
    os.makedirs("assets/output", exist_ok=True)
    save_image_bytes(output_bytes, "assets/output/output.png")
