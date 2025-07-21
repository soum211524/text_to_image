import torch
from diffusers import StableDiffusionXLPipeline
import gradio as gr


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")


def generate_image(prompt):
    if not prompt.strip():
        return None
    image = pipe(prompt).images[0]
    return image


gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt here..."),
    outputs="image",
    title="The Text-to-Image Generator",
    description="Generate stunning AI images using Stable Diffusion XL. Try prompts like 'a man driving a car on a mountain road at sunset'."
).launch()
