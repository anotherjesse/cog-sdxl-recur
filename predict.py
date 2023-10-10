from diffusers import StableDiffusionXLImg2ImgPipeline
from typing import Iterator

import os
import shutil
import subprocess
import time

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils import load_image


def scale_image(img, scale):
    width, height = img.size

    new_width = int(width * scale)
    new_height = int(height * scale)

    zoomed_img = img.resize((new_width, new_height))

    # FIXME(ja): handle scale < 1
    if scale < 1:
        return zoomed_img

    left = (zoomed_img.width - width) / 2
    top = (zoomed_img.height - height) / 2
    right = (zoomed_img.width + width) / 2
    bottom = (zoomed_img.height + height) / 2

    zoomed_img = zoomed_img.crop((left, top, right, bottom))

    return zoomed_img


SDXL_MODEL_CACHE = "./sdxl-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.txt2img_pipe.watermark = None
        self.txt2img_pipe.to("cuda")

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.watermark = None
        self.img2img_pipe.to("cuda")

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="initial image (optional)",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "KarrasDPM",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "PNDM",
            ],
            default="K_EULER",
        ),
        steps: int = Input(
            description="Number of denoising steps per image", ge=1, le=500, default=10
        ),
        frames: int = Input(description="Number of frames", ge=1, le=500, default=20),
        zoom: float = Input(description="Zoom factor", ge=1.0, le=4.0, default=1.05),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        if image:
            image = self.load_image(image)

        args = {
            "prompt": [prompt],
            "negative_prompt": [negative_prompt],
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": steps,
        }


        for frame in range(frames):
            if image:
                args["image"] = image
                args["strength"] = prompt_strength
                del args["width"]
                del args["height"]
                pipe = self.img2img_pipe
            else:
                args["width"] = width
                args["height"] = height
                pipe = self.txt2img_pipe

            pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

            image = pipe(**args).images[0]
            fn = f"{frame:04d}.jpg"
            image.save(fn)
            yield Path(fn)

            if zoom:
                image = scale_image(image, zoom)
