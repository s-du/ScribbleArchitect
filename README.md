![example](docs/scribble_banner2.png)

>**Line drawing, style transfer and upscale with Stable Diffusion!**

This GUI allows generating images from simple brush strokes, or Bezier curves, in realtime. The functions have been designed primarily for use in architecture, and for sketching in the early stages of a project. It uses Stable Diffusion and ControlNet as AI backbone for the generative process. IP Adapter support is included, as well as a large library of predefined styles! Each reference image allows to transfer a specific style to your black and white line work. An upscale function was also added, to export results in high resolution. It uses a ControlNet upscaler.

<p align="center">
    <img src="docs/tablet2.png" width="600" alt="Tablet support" style="display: block; margin: auto auto;">
</p>



Many new functions added:
- Tablet drawing support (touch screen, Ipad, ...)
- Custom style import
- Draw over background model
- Import image (and edge detection as input)
- ...

<p align="center">
    <img src="docs/anim2.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">example showing live drawing</i>
    </p>
</p>

<p align="center">
    <img src="docs/anim1.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">example showing the change of style (interior)</i>
    </p>
</p>

<p align="center">
    <img src="docs/anim4.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">example showing the change of style (exterior) </i>
    </p>
</p>

## Installation
- Install CUDA (if not done already)
- Clone the repo and install a venv.
- Install torch. Example for CUDA 11.8:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
 (see https://pytorch.org/get-started/locally/)
- Install other dependencies (see requirements):
    - accelerate
    - diffusers
    - transformers
    - PyQT6
    - opencv-python
    - opencv-contrib-python
    - peft
- Launch main.py (the first launch can be long due to the models installation process!)

## Usage
Choose a 'type' of architectural design (exterior render, facade elevation, interior render, ...) and a style. On the left zone, paint with a brush and see the proposed image adapting live (a checkbox allows to disable live inference). If you lack inspiration, or for testing purpose, a example line drawing can be generated automatically.
Mouse wheel to adapt cursor size. 

We added a screen capture function. it creates a capture box (blue border) that can be dragged around. Once happy with the capture, click again on the tool to desactivate it. It allows to work with powerful tools as input (Adobe Illustrator, Inkscape, ...).

<p align="center">
    <img src="docs/anim5.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">Screen Capture function with Inkscape as input </i>
    </p>
</p>

## Upscaling
The render can be exported in high resolution, thanks to a ControlNet upscaler. More options will be integrated soon!
<p align="center">
    <img src="docs/anim7.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">Upscaling (artistic exterior render) </i>
    </p>
</p>

<p align="center">
    <img src="docs/anim9.gif" width="800" alt="Description" style="display: block; margin: 0 auto;">
    <p align="center">
    <i style="display: block; margin-top: 5px;">Upscaling (realistic interior render) </i>
    </p>
</p>

## Tablet support
Support for drawing media has recently be included. The pressure of the pen should be detected. Tested on Ipad Pro + EasyCanvas.

https://github.com/s-du/ScribbleArchitect/assets/53427781/d827b763-f7b4-4e1d-b0e7-1f628a62b924



## Options
The SD model can be adapted in the lcm.py file. Live drawing requires a strong GPU, I would advice to reduce image size (in main.py) if too laggy! Image upscale is really GPU intensive...

## Included models
By default, the app uses Dreamshaper (https://huggingface.co/Lykon/dreamshaper-8)

![compil](docs/s_archi_compil.png)
