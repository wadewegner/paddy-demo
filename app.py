import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize an empty list to store image-caption pairs
gallery = []

def process_image(image):
    # Convert to RGB if the image is in RGBA format
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Add new image-caption pair to the beginning of the gallery
    gallery.insert(0, (image, caption))
    
    # Return the updated gallery
    return gallery

# Define the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"]),
    outputs=gr.Gallery(label="Generated Captions", columns=2),
    title="Image Captioning with BLIP",
    description="Upload an image or take a picture, and the BLIP model will generate a caption for it.",
)

# Launch the app
iface.launch()
