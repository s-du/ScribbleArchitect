import torch
from RealESRGAN import RealESRGAN

import time
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from PIL import Image, ImageEnhance
import cv2
import numpy as np

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
}

SD15_WEIGHTS = "weights"
CONTROLNET_CACHE = "controlnet-cache"


def super_tile_upscale(source_image, prompt, choice_resolution, hdr_choice=0.5):
    def resize_for_condition_image(input_image, resolution):
        scale = 2
        if resolution == 2048:
            init_w = 1024
        elif resolution == 2560:
            init_w = 1280
        elif resolution == 3072:
            init_w = 1536
        else:
            init_w = 1024
            scale = 4

        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(init_w) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        model = ESRGAN_models[scale]
        img = model.predict(img)
        return img

    def calculate_brightness_factors(hdr_intensity):
        factors = [1.0] * 9
        if hdr_intensity > 0:
            factors = [1.0 - 0.9 * hdr_intensity, 1.0 - 0.7 * hdr_intensity, 1.0 - 0.45 * hdr_intensity,
                       1.0 - 0.25 * hdr_intensity, 1.0, 1.0 + 0.2 * hdr_intensity,
                       1.0 + 0.4 * hdr_intensity, 1.0 + 0.6 * hdr_intensity, 1.0 + 0.8 * hdr_intensity]
        return factors

    def pil_to_cv(pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def adjust_brightness(cv_image, factor):
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        v = np.clip(v * factor, 0, 255).astype('uint8')
        adjusted_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    def create_hdr_effect(original_image, hdr):
        # Convert PIL image to OpenCV format
        cv_original = pil_to_cv(original_image)

        brightness_factors = calculate_brightness_factors(hdr)
        images = [adjust_brightness(cv_original, factor) for factor in brightness_factors]

        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(images)
        hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
        hdr_image_pil = Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

        return hdr_image_pil

    """Load the model into memory to make running multiple predictions efficient"""

    print("Loading pipeline...")
    st = time.time()

    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1e_sd15_tile',
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        torch_dtype=torch.float16,
        controlnet=controlnet
    ).to("cuda")

    ESRGAN_models = {}

    for scale in [2, 4]:
        ESRGAN_models[scale] = RealESRGAN("cuda", scale=scale)
        ESRGAN_models[scale].load_weights(
            f"models/upscale/RealESRGAN_x{scale}.pth", download=False
        )

    print("Setup complete in %f" % (time.time() - st))

    print("Start prediction")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(425)
    loaded_image = source_image.convert("RGB")
    control_image = resize_for_condition_image(loaded_image, choice_resolution)
    final_image = create_hdr_effect(control_image, hdr_choice)

    args = {
        "prompt": prompt,
        "image": final_image,
        "control_image": final_image,
        "strength": 0.7,
        "controlnet_conditioning_scale": 0.7,
        "negative_prompt": 'low quality, bad result',
        "guidance_scale": 7,
        "generator": generator,
        "num_inference_steps": 20,
        "guess_mode": False,
    }

    w, h = control_image.size

    if (w * h > 2560 * 2560):
        pipe.enable_vae_tiling()
    else:
        pipe.disable_vae_tiling()

    output = pipe(**args).images[0]

    return output