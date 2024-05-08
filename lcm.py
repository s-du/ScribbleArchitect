import os
import random
from os import path
from contextlib import nullcontext
import time
from sys import platform
import torch
import cv2
import resources as res
import torch
from PIL import Image
import numpy as np


"""
All credits to https://github.com/flowtyone/flowty-realtime-lcm-canvas!!
"""

cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
is_mac = platform == "darwin"

model_list = ['Dreamshaper7', 'SD 1.5','Dreamshaper8','AbsoluteReality', 'RevAnimated', 'Protogen',  'SDXL 1.0']
model_ids = [ "Lykon/dreamshaper-7", "runwayml/stable-diffusion-v1-5", "Lykon/dreamshaper-8","Lykon/absolute-reality-1.81", "danbrown/RevAnimated-v1-2-2", "darkstorm2150/Protogen_x5.8_Official_Release", "stabilityai/stable-diffusion-xl-base-1.0"]


def screen_to_lines(image,option):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if option == 0:
        gray_image_bil = cv2.bilateralFilter(gray_image, 5, 75, 75)

        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray_image_bil, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image_bil, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2), 0.5, cv2.pow(sobely, 2), 0.5, 0))

        # Normalize and convert to 8-bit format
        sobel_image = cv2.convertScaleAbs(sobel_image)

        # Invert the Sobel image
        processed_image = 255 - sobel_image

    elif option == 1:
        # Apply a bilateral filter
        gray_image_bil = cv2.bilateralFilter(gray_image, 5, 75, 75)

        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray_image_bil, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image_bil, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2), 0.5, cv2.pow(sobely, 2), 0.5, 0))

        # Normalize and convert to 8-bit format
        sobel_image = cv2.convertScaleAbs(sobel_magnitude)

        # Invert the Sobel image to use dark areas as thicker lines
        processed_image = 255 - sobel_image

        # Threshold to create binary image
        _, binary_image = cv2.threshold(processed_image, 200, 255, cv2.THRESH_BINARY)

        # Optional: Use dilation to thicken the darker lines
        kernel = np.ones((2, 2), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)

        processed_image=binary_image


    elif option == 2:
        edges = cv2.Canny(gray_image, 100, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 3:
        edges = cv2.Canny(gray_image, 100, 200, L2gradient=True)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 4:
        gray_image_bil = cv2.bilateralFilter(gray_image, 5, 75, 75)
        edges = cv2.Canny(gray_image_bil, 100, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 5:
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 6 or option == 7:
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection('models/edge_model.yml')
        # detect the edges
        print(image.dtype, image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        edges = edge_detector.detectEdges(image)

        # Invert the Sobel image
        u_im = np.uint8(255 * edges)
        processed_image = 255 - u_im

        if option == 7:
            # Threshold to create binary image
            _, binary_image = cv2.threshold(processed_image, 200, 255, cv2.THRESH_BINARY)

            # Optional: Use dilation to thicken the darker lines
            kernel = np.ones((2, 2), np.uint8)
            binary_image = cv2.dilate(binary_image, kernel, iterations=1)

            processed_image = binary_image

    elif option == 8:
        processed_image = gray_image


    return processed_image

def should_use_fp16():
    if is_mac:
        return True

    gpu_props = torch.cuda.get_device_properties("cuda")

    if gpu_props.major < 6:
        return False

    nvidia_16_series = ["1660", "1650", "1630"]

    for x in nvidia_16_series:
        if x in gpu_props.name:
            return False

    return True

class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")


def load_models(model_id="Lykon/dreamshaper-8", use_ip=True):
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline
    from diffusers.utils import load_image

    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    ip_adapter_name = "ip-adapter_sd15.bin"
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        cache_dir=cache_path,
        controlnet=controlnet,
        controlnet_conditioning_scale=0.9,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    )

    if use_ip:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_adapter_name)

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm_lora_id)
    pipe.fuse_lora()

    device = "mps" if is_mac else "cuda"

    pipe.to(device=device)

    generator = torch.Generator()

    def infer(
            prompt,
            negative_prompt,
            image,
            num_inference_steps=4,
            guidance_scale=1,
            strength=0.9,
            seed=random.randrange(0, 2**63),
            ip_scale=0.8,
            ip_image_to_use='',
            cn_strength=0.8,
    ):

        with torch.inference_mode():
            with torch.autocast("cuda") if device == "cuda" else nullcontext():
                with timer("inference"):
                    if use_ip:
                        pipe.set_ip_adapter_scale(ip_scale)
                        return pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=load_image(image),
                            ip_adapter_image=load_image(ip_image_to_use),
                            generator=generator.manual_seed(seed),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=strength,
                            controlnet_conditioning_scale=cn_strength
                        ).images[0]
                    else:
                        return pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=load_image(image),
                            generator=generator.manual_seed(seed),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=strength
                        ).images[0]

    return infer


def tile_upscale(source_image, prompt, res):
    from diffusers import ControlNetModel, DiffusionPipeline
    from diffusers.utils import load_image
    def resize_for_condition_image(input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile',
                                                 torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained("Lykon/dreamshaper-8",
                                             custom_pipeline="stable_diffusion_controlnet_img2img",
                                             controlnet=controlnet,
                                             torch_dtype=torch.float16,
                                             variant="fp16").to('cuda')

    condition_image = resize_for_condition_image(source_image, res)

    image = pipe(prompt="best quality, "+ prompt,
                 negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                 image=condition_image,
                 controlnet_conditioning_image=condition_image,
                 width=condition_image.size[0],
                 height=condition_image.size[1],
                 strength=1.0,
                 generator=torch.manual_seed(0),
                 num_inference_steps=32,
                 ).images[0]

    return image
