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
        prompt: str = Input(description="text to generate the image from", default=""),
        negative_prompt: str = Input(description="Text describing image traits to avoid during generation", default=""),
        guidance_scale: float = Input(description="Floating-point number represeting how closely to adhere to prompt description.", ge=1, le=50, default=12),
        image_encoding:str = Input(description="Define which encoding process should be applied before returning the generated image(s).", default="jpeg"),
        steps: int = Input(description="Integer representing how many steps of diffusion to run", ge=1, default=50),
        height: int = Input(description="The height in pixels of the generated image", default=512),
        width: int = Input(description="The width in pixels of the generated image", default=512),
        controlnet_image: str = Input(description="Controlnet image encoded in b64 string for guiding image generation", default=""),
    ) -> bytes:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
        negative_prompt = "low quality, bad quality, sketches"
        controlnet_conditioning_scale = 0.5

        if controlnet_image == "":
            image = load_image(
                "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            )
            # recommended for good generalization
            # get canny image
            image = np. array(image)
            image = cv2. Canny(image, 100, 200)
            image = image[:, :, None]
            image = np. concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)
        else:
            canny_image = Image.open(BytesIO(base64.b64decode(controlnet_image)))

        # generate images
        image = self.pipe(
            prompt, 
            image=canny_image,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            image_encoding=image_encoding,
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format=image_encoding)
        img_str = base64.b64encode(buffered.getvalue())

        return img_str
