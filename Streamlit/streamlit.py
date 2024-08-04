import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import requests
import numpy as np
from matplotlib import pyplot as plt

st.image('./images/logos/Both.png', width=500)

st.title("Image Captioning & Search Engine 🖼️🔍")


blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

urls = [
    "https://picsum.photos/id/17/2500/1667",
    "https://picsum.photos/id/5/5000/3334",
    "https://picsum.photos/id/57/2448/3264",
    "https://picsum.photos/id/74/4288/2848",
    "https://picsum.photos/id/133/2742/1828",
    "https://picsum.photos/id/197/4272/2848",
    "https://picsum.photos/id/219/5000/3333",
    "https://picsum.photos/id/274/3824/2520",
    "https://picsum.photos/id/292/3852/2556",
    "https://picsum.photos/id/364/5000/2917",
    "https://picsum.photos/id/454/4403/2476",
    "https://picsum.photos/id/509/4608/3456",
    "https://picsum.photos/id/584/2507/1674",
    "https://picsum.photos/id/824/5000/3333",
    "https://picsum.photos/id/841/5000/2522"
]

images = [Image.open(requests.get(url, stream=True).raw) for url in urls]


st.header("Search Engine 🔍")

clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

search_caption = st.text_input("Enter a caption to search for an image 🖼️🔍:")

if st.button("Search"):
    inputs = clip_processor(
        text=[search_caption], images=images,
        return_tensors='pt', padding=True
    )

    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.argmax()
    st.image(images[probs.item()].resize((700, 500), Image.Resampling.BILINEAR), caption=f"Search result for: {search_caption}")
st.header("Check out these random images to see how the model captions them! 📸✨")

cols = st.columns(5)

for i, image in enumerate(images):
    cols[i % 5].image(image, width=300)

st.header("Upload an Image to Get a Caption 🖼️📋")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
        
    st.write("Generating caption...")
    
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    
    st.write(f"Caption: {caption}")


