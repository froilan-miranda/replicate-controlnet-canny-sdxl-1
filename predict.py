# Prediction interface for Cog ⚙️
# https://cog.run/python



import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionXLControlNetPipeline)
from diffusers.utils import load_image
from PIL import Image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        )
        self.pipe. enable_model_cpu_offload()

    def predict(
        self,
        prompt: str = Input(description="text to generate the image from"),
        negative_prompt: str = Input(description="Text describing image traits to avoid during generation"),
        controlnet_image: str = Input(description="Controlnet image encoded in b64 string for guiding image generation"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=1.5
        ),
    ) -> bytes:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        image = load_image(
            "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
        )
        prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
        negative_prompt = "low quality, bad quality, sketches"
        controlnet_conditioning_scale = 0.5

        # recommended for good generalization
        # get canny image
        image = np. array(image)
        image = cv2. Canny(image, 100, 200)
        image = image[:, :, None]
        image = np. concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        # generate images
        image = self.pipe(
            prompt, 
            image=canny_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str
