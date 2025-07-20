# text_to_image
🖼️ AI Image Generator with Stable Diffusion XL This project is a simple, GPU-powered Text-to-Image Generator built using Stable Diffusion XL (SDXL), the latest model from Stability AI, and powered by Gradio for a fast and interactive web UI.
#  AI Text-to-Image Generator using Stable Diffusion XL

This project is a simple yet powerful AI-powered image generator using the [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model. Just enter a prompt and generate high-quality images instantly through an interactive Gradio web interface.

## 📦 Features

- 🔥 Based on **Stable Diffusion XL 1.0** — the latest high-resolution model
- 🧠 Uses **Hugging Face Diffusers** and **PyTorch**
- ⚡ Fast GPU inference with `fp16`
- 🌐 Interactive **Gradio** web UI

## 🛠️ Installation

Make sure you have:

- Python 3.8+
- NVIDIA GPU with CUDA (for best performance)

Then follow these steps:

```bash
git clone https://github.com/your-username/ai-image-generator.git
cd ai-image-generator
pip install -r requirements.txt
