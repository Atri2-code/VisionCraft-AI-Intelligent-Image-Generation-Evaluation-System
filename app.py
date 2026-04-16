import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

from config import DEVICE, NUM_IMAGES, MODEL_ID
from utils import enhance_prompt, generate_feedback

# ------------------ DEVICE ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD MODELS ------------------
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
pipe = pipe.to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ------------------ FUNCTIONS ------------------

def generate_images(prompt, num_images):
    images = []
    for _ in range(num_images):
        image = pipe(prompt).images[0]
        images.append(image)
    return images

def evaluate_images(images, prompt):
    inputs = clip_processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True
    )

    outputs = clip_model(**inputs)
    scores = outputs.logits_per_image.softmax(dim=1)

    return scores.detach().cpu().numpy().flatten().tolist()

def run_pipeline(prompt):
    enhanced = enhance_prompt(prompt)

    images = generate_images(enhanced, NUM_IMAGES)
    scores = evaluate_images(images, enhanced)

    best_idx = scores.index(max(scores))
    best_image = images[best_idx]

    feedback = generate_feedback(scores)

    return enhanced, images, str(scores), best_image, feedback

# ------------------ UI ------------------

interface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(label="Enter Prompt"),
    outputs=[
        gr.Textbox(label="Enhanced Prompt"),
        gr.Gallery(label="Generated Images"),
        gr.Textbox(label="Scores"),
        gr.Image(label="Best Image"),
        gr.Textbox(label="Feedback")
    ],
    title="AI Design Copilot"
)

interface.launch()
