import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
from model_arch import DeepShadowModel # Make sure this matches your architecture file name!

# 1. MAKE IT WIDE AND SLEEK
st.set_page_config(page_title="Deep Shadow Volumetry", page_icon="🛢️", layout="wide")

st.title("🛢️ Deep-Shadow Volumetry OSINT Dashboard")
st.markdown("Estimate floating-roof oil tank volumes using satellite imagery and Sim2Real AI.")

# --- Load Model Function ---
@st.cache_resource
def load_ai_model():
    model = DeepShadowModel()
    # Loading the NEW reality-adapted brain!
    model.load_state_dict(torch.load('transfer_learned_weights.pth', map_location='cpu'))
    model.eval()
    return model

model = load_ai_model()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# 2. CREATE A SICK 2-COLUMN LAYOUT
col1, col2 = st.columns([1, 1]) 

with col1:
    st.subheader("1. Target Acquisition")
    uploaded_file = st.file_uploader("Upload Satellite Crop (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    st.subheader("2. Environmental Metadata")
    elevation = st.slider("☀️ Sun Elevation Angle (°)", 10, 90, 45, help="Estimated sun angle when the photo was taken.")

    st.subheader("3. Image Processing")
    apply_filter = st.checkbox("Apply Sim2Real Contrast Filter", value=True)

with col2:
    st.subheader("Analysis & Intelligence")
    
    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert('L')
        
        if apply_filter:
            enhancer = ImageEnhance.Contrast(raw_img)
            display_img = enhancer.enhance(2.0)
        else:
            display_img = raw_img
            
        st.image(display_img, caption="Processed Network Input", use_column_width=True)
        
        # Run Inference
        if st.button("🚀 Execute Volume Analysis", type="primary", use_container_width=True):
            with st.spinner("Neural Network processing geometric shadows..."):
                input_tensor = transform(display_img).unsqueeze(0)
                meta_tensor = torch.tensor([[elevation / 90.0]], dtype=torch.float32)
                
                with torch.no_grad():
                    prediction = model(input_tensor, meta_tensor)
                    final_volume = prediction.item() * 100
                
                st.divider()
                # 3. USE ST.METRIC FOR A MASSIVE, CLEAN NUMBER
                st.metric(label="Estimated Tank Volume", value=f"{final_volume:.1f}%")
                st.progress(int(final_volume) / 100)
    else:
        st.info("Awaiting satellite telemetry... Upload an image to begin.")
