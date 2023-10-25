import openai
import base64
from diffusers import StableDiffusionPipeline
import torch

openai.api_key_path = "thesis.env"

prompts = []

def generate_dalle_image(text_prompt):
    response = openai.Image.create(
        prompt=text_prompt,
        n=1,
        size="1024x1024",
        response_format= "b64_json"
    )
    image_data = response['data'][0]['b64_json']
    image = base64.b64decode(image_data)
    filename = text_prompt.replace(" ", "_") + ".png"
    with open(filename, 'wb') as f:
        f.write(image)

def generate_dalle_dataset():
    for i in range(prompts):
        generate_dalle_image(prompts[i])

# To generate multiple images, pass prompts List to pipe()
def generate_stable_image(text_prompt):
    # alt. model runwayml/stable-diffusion-v1-5, play with different models?
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") # add torch_dtype=torch.float16 for gpu
    pipe = pipe.to("cpu") # change to cuda for gpu
    image = pipe(text_prompt).images[0]
    filename = text_prompt.replace(" ", "_") + ".png"
    image.save(filename)

generate_stable_image("oaks in the mountains of carrara in the style of realism")
generate_dalle_image("oaks in the mountains of carrara in the style of realism")

