"""
Author: SDU (sdu@bbri.be)
Part of the code was inspired from https://github.com/flowtyone/flowty-realtime-lcm-canvas!!
"""
import os
import random
import time
from os import path
from contextlib import nullcontext
from sys import platform

import torch
from torch import nn
import cv2
import numpy as np
from PIL import Image


cache_path = path.join(path.dirname(path.abspath(__file__)), "models")

os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
is_mac = platform == "darwin"

model_list = ['Dreamshaper8']
model_ids = ["Lykon/dreamshaper-8"]

LINE_METHODS = ['Sobel Custom', 'Canny', 'Canny + L2', 'Canny + BIL', 'Canny + Blur', 'RF Custom']


PALETTE = palette = np.asarray([
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])


def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Sort the images if needed

    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 videos
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def otsu_threshold(image):
    # Convert image to numpy array
    pixel_array = np.array(image)

    # Compute histogram
    histogram, bin_edges = np.histogram(pixel_array, bins=256, range=(0, 255))

    # Normalize histogram
    histogram = histogram / np.sum(histogram)

    # Compute cumulative sums and means
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    # Compute global mean
    global_mean = cumulative_mean[-1]

    # Compute between-class variance for each threshold
    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (
                cumulative_sum * (1 - cumulative_sum))
    between_class_variance = np.nan_to_num(between_class_variance)  # Handle division by zero

    # Find the threshold that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    return optimal_threshold
def img_to_seg(image):
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    # Load the pre-trained models
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

    # Determine the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process the image and move the tensor to the appropriate device
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)

    # Move the model to the same device
    model = model.to(device)

    # Perform the forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Get the logits and move them to the appropriate device
    logits = outputs.logits

    # Use the shape attribute of the numpy array to get height and width
    height, width = image.shape[:2]

    # Upsample the logits to the original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    # Get the predicted segmentation map
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu()  # Move the result to the CPU

    # Create a color segmentation image
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(PALETTE):
        color_seg[pred_seg == label, :] = color
    # color_seg = color_seg[..., ::-1]  # Convert RGB to BGR

    return color_seg


def screen_to_lines(image, option):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
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
    """
    if option == 0:
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

        processed_image = binary_image


    elif option == 1:
        edges = cv2.Canny(gray_image, 100, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 2:
        edges = cv2.Canny(gray_image, 100, 200, L2gradient=True)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 3:
        gray_image_bil = cv2.bilateralFilter(gray_image, 5, 75, 75)
        edges = cv2.Canny(gray_image_bil, 100, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 4:
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)

        # Invert the Sobel image
        processed_image = 255 - edges

    elif option == 5:
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection('models/edge_model.yml')
        # detect the edges
        print(image.dtype, image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        edges = edge_detector.detectEdges(image)

        # Invert the Sobel image
        u_im = np.uint8(255 * edges)
        processed_image = 255 - u_im

        # Threshold to create binary image
        _, binary_image = cv2.threshold(processed_image, 200, 255, cv2.THRESH_BINARY)

        # Optional: Use dilation to thicken the darker lines
        kernel = np.ones((2, 2), np.uint8)
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)

        processed_image = binary_image

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


def load_models_img2img(model_id="Lykon/dreamshaper-8", use_ip=True):
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline, \
        StableDiffusionControlNetImg2ImgPipeline
    from diffusers.utils import load_image

    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    ip_adapter_name = "ip-adapter_sd15.bin"
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
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
            cn_image,
            num_inference_steps=4,
            guidance_scale=1,
            strength=0.75,
            seed=random.randrange(0, 2 ** 63),
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
                            control_image=load_image(cn_image),
                            ip_adapter_image=load_image(ip_image_to_use),
                            generator=generator.manual_seed(seed),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=cn_strength,
                            controlnet_conditioning_scale=0.9
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


def load_models(model_id="Lykon/dreamshaper-8", use_ip=True):
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline, \
        StableDiffusionControlNetImg2ImgPipeline
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
            seed=random.randrange(0, 2 ** 63),
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


def load_models_multiple_cn(model_id="Lykon/dreamshaper-8", use_ip=True):
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline
    from diffusers.utils import load_image

    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    ip_adapter_name = "ip-adapter_sd15.bin"
    controlnets = [
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)]

    if 'custom' in model_id:
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            model_id,
            cache_dir=cache_path,
            controlnet=controlnets,
            controlnet_conditioning_scale=[0.9,0.9],
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            cache_dir=cache_path,
            controlnet=controlnets,
            controlnet_conditioning_scale=[0.9,0.9],
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
            images,
            num_inference_steps=4,
            guidance_scale=1,
            strength=0.9,
            seed=random.randrange(0, 2 ** 63),
            ip_scale=0.8,
            ip_image_to_use='',
            cn_strength=[0.9, 0.9],
    ):

        with torch.inference_mode():
            with torch.autocast("cuda") if device == "cuda" else nullcontext():
                with timer("inference"):
                    if use_ip:
                        pipe.set_ip_adapter_scale(ip_scale)
                        return pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=images,
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
                            image=load_image(images),
                            generator=generator.manual_seed(seed),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=strength
                        ).images[0]

    return infer


def standard_upscale(source_image, prompt):
    from diffusers import StableDiffusionUpscalePipeline

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")

    upscaled_image = pipeline(prompt=prompt, image=source_image).images[0]

    return upscaled_image


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

    image = pipe(prompt='highest quality',
                 negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality",
                 image=condition_image,
                 controlnet_conditioning_image=condition_image,
                 width=condition_image.size[0],
                 height=condition_image.size[1],
                 strength=1,
                 generator=torch.manual_seed(0),
                 num_inference_steps=40,
                 ).images[0]

    return image
