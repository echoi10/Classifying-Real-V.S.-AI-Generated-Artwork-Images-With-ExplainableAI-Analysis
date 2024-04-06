import openai
import base64
from diffusers import StableDiffusionPipeline
import torch

# To generate multiple images, pass prompts List to pipe()
def generate_stable_image(text_prompt):
    # alt. model runwayml/stable-diffusion-v1-5, play with different models?
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") # add torch_dtype=torch.float16 for gpu
    pipe = pipe.to("cpu") # change to cuda for gpu
    image = pipe(text_prompt).images[0]
    filename = text_prompt.replace(" ", "_") + ".png"
    image.save(filename)

generate_stable_image("oaks in the mountains of carrara in the style of realism")

