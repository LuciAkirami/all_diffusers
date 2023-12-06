import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(device)
    # pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
    return pipe

pipe = load_model()
text = st.text_input("Enter your text")

if st.sidebar.button('Generate Image'):
    with st.spinner('Generating image...'):
        # Generate the image
        generated_image = pipe(prompt=text, num_inference_steps=1, guidance_scale=0.0).images[0]

        st.image(generated_image)