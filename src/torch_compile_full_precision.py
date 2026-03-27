from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import modal
import torch
import torch.nn as nn

ckpts_vol = modal.Volume.from_name("flux2_ckpts", create_if_missing=True)
inductor_vol = modal.Volume.from_name("inductor_aot_models", create_if_missing=True)
inductor_cache_vol = modal.Volume.from_name("inductor-cache", create_if_missing=True)
nv_cache_vol = modal.Volume.from_name("nv-cache", create_if_missing=True)
triton_cache_vol = modal.Volume.from_name("triton-cache", create_if_missing=True)
ckpts_path = "/checkpoints"
aot_path = "/artifacts"

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:26.02-py3",
    )
    .uv_pip_install(
        "git+https://github.com/BhagyeshKothalkar/flux2",
        extra_index_url="https://pypi.nvidia.com",
    )
    .env(
        {
            "HF_HUB_CACHE": ckpts_path,
            "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
            "TRITON_CACHE_DIR": "/root/.triton",
            "CUDA_CACHE_PATH": "/root/.nv",
        }
    )
)

app = modal.App("flux2-klein")


@dataclass
class Config:
    prompt: str = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 4
    guidance: float = 2.8
    input_images: List[Path] = field(default_factory=list)
    match_image_size: Optional[int] = None  # Index of input_images to match size from
    upsample_prompt_mode: Literal["none", "local", "openrouter"] = "none"
    openrouter_model: str = "mistralai/pixtral-large-2411"  # OpenRouter model name

    def copy(self) -> "Config":
        return Config(
            prompt=self.prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            num_steps=self.num_steps,
            guidance=self.guidance,
            input_images=list(self.input_images),
            match_image_size=self.match_image_size,
            upsample_prompt_mode=self.upsample_prompt_mode,
            openrouter_model=self.openrouter_model,
        )


class wrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        x,
        x_ids,
        timesteps,
        ctx,
        ctx_ids,
        guidance,
    ):

        return self.transformer(
            x,
            x_ids.contiguous().to(torch.bfloat16),
            timesteps.contiguous().to(torch.bfloat16),
            ctx.contiguous().to(torch.bfloat16),
            ctx_ids.contiguous().to(torch.bfloat16),
            guidance.contiguous().to(torch.bfloat16),
        )


DEFAULTS = Config()


def img_to_b64_string(image):
    import base64
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    im_data = buffer.getvalue()
    im_b64 = base64.b64encode(im_data)
    return im_b64.decode(encoding="utf-8")


def save_image_bytes(img_bytes, save_name="output.png"):
    with open(save_name, "wb") as f:
        f.write(img_bytes)
    print(f"saved image to {save_name}")


gpu = "L40S"
compilation_suffix = "O3"
safe_model_name = "flux_2_klein_9b"


@app.cls(
    image=image,
    gpu=gpu,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        ckpts_path: ckpts_vol,
        aot_path: inductor_vol,
        "/root/.nv": nv_cache_vol,
        "/root/.triton": triton_cache_vol,
        "/root/.inductor-cache": inductor_cache_vol,
    },
)
class FluxRun:
    @modal.enter()
    def enter(self):

        import os

        from flux2.util import FLUX2_MODEL_INFO, load_ae, load_text_encoder

        # config = FLUX2_MODEL_INFO
        # config.update(fp8_entry)
        model_name = "flux.2-klein-9b"
        self.model_info = FLUX2_MODEL_INFO[model_name]

        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")

        self.package_path = os.path.join(
            aot_path,
            safe_model_name,
            f"{safe_model_name}_{gpu}_{compilation_suffix}.pt2",
        )
        self.text_encoder = load_text_encoder(model_name, self.device)
        self.loaded_model = wrapper(
            torch._inductor.aoti_load_package(self.package_path)
        )
        self.ae = load_ae(model_name)
        self.loaded_model.eval()
        self.ae.eval()
        self.text_encoder.eval()

        self.cfg = DEFAULTS.copy()

        defaults = self.model_info.get("defaults", {})
        self.cfg.num_steps = defaults["num_steps"]
        self.cfg.guidance = defaults["guidance"]

    @modal.method()
    def infer(self, prompt: str, cond_image_b64: str):

        import time

        t0 = time.perf_counter()
        batch_size = 2

        import base64
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

        if not prompt or prompt.strip() == "":
            prompt = "A high-quality image"

        cond_image_bytes = base64.b64decode(cond_image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(cond_image_bytes)
            tmp.flush()
            inp_path = tmp.name

            if prompt is not None:
                self.cfg.prompt = prompt
            height = self.cfg.height
            width = self.cfg.width
            img = Image.open(inp_path)
            img_ctx: List[Image.Image] = [img, img]
            prompt_batch = [prompt, prompt]
            seed = (
                self.cfg.seed if self.cfg.seed is not None else random.randrange(2**31)
            )

            t1 = time.perf_counter()
            print(f"random configuring took {t1 - t0}")
            with torch.no_grad():
                ref_tokens, ref_ids = encode_image_refs(self.ae, img_ctx)
                if ref_tokens is not None and ref_ids is not None:
                    if ref_tokens.shape[0] != batch_size:
                        ref_tokens = ref_tokens.expand(batch_size, -1, -1).contiguous()
                        ref_ids = ref_ids.expand(batch_size, -1, -1).contiguous()
                ctx = self.text_encoder(prompt_batch).to(torch.bfloat16)
                ctx, ctx_ids = batched_prc_txt(ctx)

                shape = (2, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(42)
                randn = torch.randn(
                    shape, generator=generator, dtype=torch.bfloat16, device="cuda"
                )
                x, x_ids = batched_prc_img(randn)
                timesteps = get_schedule(self.cfg.num_steps, x.shape[1])
                t2 = time.perf_counter()
                print(f"pre-denoising things took {t2 - t1}")
                denoise_fn = denoise
                x = denoise_fn(
                    self.loaded_model,  # type:ignore
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
                print(f"denoising took {t3 - t2}")
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = self.ae.decode(x).float()
                x = x.clamp(-1, 1)
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

                image_b64 = img_to_b64_string(img)
                t4 = time.perf_counter()
                print(f"generation took {t4 - t0}")
                return image_b64


"""
x.shape torch.Size([1, 4096, 128])
x_ids.shape torch.Size([1, 4096, 4])
ctx.shape torch.Size([1, 512, 12288])
ctx_ids.shape torch.Size([1, 512, 4])
timesteps [1.0, 0.9673840403556824, 0.9081438779830933, 0.7671999335289001, 0.0]
self.cfg.guidance 1.0
ref_tokens.shape torch.Size([1, 4096, 128])
ref_ids.shape torch.Size([1, 4096, 4])
"""


@app.cls(
    image=image,
    gpu=gpu,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        ckpts_path: ckpts_vol,
        aot_path: inductor_vol,
        "/root/.nv": nv_cache_vol,
        "/root/.triton": triton_cache_vol,
        "/root/.inductor-cache": inductor_cache_vol,
    },
    timeout=3000,
)
class Compiler:
    @modal.enter()
    def enter(self):
        pass

        import multiprocessing
        import os

        import torch._inductor.config as inductor_config
        from flux2.util import load_flow_model

        model_name = "flux.2-klein-9b"
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")
        self.model = load_flow_model(model_name, device=device)
        ckpts_vol.commit()
        self.model.eval()
        b = 2
        x = torch.rand((b, 4096, 128), device=device, dtype=torch.bfloat16)
        x_ids = torch.rand((b, 4096, 4), device=device, dtype=torch.bfloat16)
        ctx = torch.rand((b, 512, 12288), device=device, dtype=torch.bfloat16)
        ctx_ids = torch.rand((b, 512, 4), device=device, dtype=torch.bfloat16)
        timesteps = torch.rand((b,), device=device, dtype=torch.bfloat16)
        guidance = torch.full((b,), 1.0, device=device, dtype=torch.bfloat16)
        ref_tokens = torch.rand((b, 4096, 128), device=device, dtype=torch.bfloat16)
        ref_ids = torch.rand((b, 4096, 4), device=device, dtype=torch.bfloat16)
        x = torch.cat((x, ref_tokens), dim=1)
        x_ids = torch.cat((x_ids, ref_ids), dim=1)
        self.dummy_args = (
            x,
            x_ids,
            timesteps,
            ctx,
            ctx_ids,
            guidance,
        )
        self.package_path = os.path.join(
            aot_path,
            safe_model_name,
            f"{safe_model_name}_{gpu}_{compilation_suffix}.pt2",
        )
        os.makedirs(os.path.dirname(self.package_path), exist_ok=True)
        torch.set_float32_matmul_precision("high")

        inductor_config.compile_threads = multiprocessing.cpu_count()
        inductor_config.fx_graph_cache = True
        inductor_config.autotune_local_cache = True
        inductor_config.disable_progress = False
        inductor_config.max_autotune = True  # Exhaustive kernel search
        inductor_config.freezing = True  # CONSTANT FOLDING (Crucial for AOT inference)
        inductor_config.coordinate_descent_tuning = (
            True  # Advanced Triton tuning heuristics
        )
        inductor_config.layout_optimization = (
            True  # Allow memory layout reordering for speed
        )
        inductor_config.triton.cudagraphs = True
        inductor_config.triton.cudagraph_trees = False

        inductor_config.aot_inductor.compile_wrapper_opt_level = (
            "O3"  # Max C++ optimization
        )
        inductor_config.cuda.enable_cuda_lto = (
            True  # Link Time Optimization for the wrapper
        )
        inductor_config.aot_inductor.emit_multi_arch_kernel = (
            False  # L40S only, save compile time/size
        )
        inductor_config.coordinate_descent_check_all_directions = True
        inductor_config.epilogue_fusion = True
        inductor_config.triton.multi_kernel = 0  # Keep at 0 to maximize kernel fusion
        inductor_config.triton.store_cubin = True
        inductor_config.aot_inductor.package = True

        # --- 4. Cloud Portability / Safety ---
        # Prevents "Illegal Instruction" CPU crashes across heterogeneous nodes
        os.environ["TORCH_INDUCTOR_CPP_VEC_ISA"] = "avx2"
        inductor_config.cpp.vec_isa_ok = False

    @modal.method()
    def compile(self):
        print("starting compilaiton")

        from torch.export import export

        with torch.no_grad():
            try:
                print("exporting...")
                exported_program = export(self.model, self.dummy_args, strict=False)
                print("AOT Compiling to .pt2 (Fast Mode)...")
                output_path = torch._inductor.aoti_compile_and_package(
                    exported_program,
                    package_path=self.package_path,
                )
                print("committing")
                inductor_vol.commit()
                print("saved to:", output_path)
                nv_cache_vol.commit()
                triton_cache_vol.commit()
                inductor_cache_vol.commit()
                print("done committing")
            except Exception as e:
                print("error  while compiling", e)
                raise e
        return output_path


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
def hi():

    import base64

    from PIL import Image

    input_image_path = "assets/input/input.png"
    prompt = edit_prompt

    input_image = Image.open(input_image_path).resize((1024, 1024))
    instance = FluxRun()
    # compilaiton = Compiler()
    # handle = compilaiton.compile.spawn()

    # try:
    #     while True:
    #         try:
    #             handle.get(timeout=30)
    #             print("Compilation complete.")
    #             break
    #         except TimeoutError:
    #             print(f"--- [Local Heartbeat] Still waiting for {handle.object_id}... ---")
    #             continue
    # except Exception as e:
    #     print(f"Task failed or timed out: {e}")

    # print("sleeping to ensure sync")
    # import time

    # time.sleep(10)

    b64_string = img_to_b64_string(input_image)
    output_b64_string = instance.infer.remote(prompt, b64_string)
    output_bytes = base64.b64decode(output_b64_string)
    save_image_bytes(output_bytes, "assets/output/output.png")
