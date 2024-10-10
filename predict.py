# Prediction interface for Cog ⚙️
# https://cog.run/python
import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input
from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                       LMSDiscreteScheduler, PNDMScheduler,
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
        controlnet_conditioning_scale: float = Input(description="The outputs of the ControlNet are multiplied by controlnet_conditioning_scale before they are added to the residual in the original unet", default=1),
        scheduler: str = Input(description="A scheduler to be used in combination with unet to denoise the encoded image latents.", defualt="DDIM"),
        control_guidance_start: float = Input(description="The percentage of total steps at which the ControlNet starts applying", defualt=0.0),
        control_guidance_end: float = Input(description="The percentage of total steps at which the ControlNet ends applying", defualt=1.0)
    ) -> str:
        if controlnet_image == "":
            prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
            negative_prompt = "low quality, bad quality, sketches"
            controlnet_conditioning_scale = 0.5
            image = load_image(
                "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
            )
            # recommended for good generalization
            # get canny image
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)
        else:
            image_data = base64.b64decode(controlnet_image)
            canny_image = Image.open(BytesIO(image_data))

        # Load Scheduler
        match scheduler:
            case "DDIM":
                self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            case "LMSDiscrete":
                self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            case "PNDM":
                self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

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
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format=image_encoding)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str
