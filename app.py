
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model_arch import DeepShadowModel

st.set_page_config(page_title="Deep-Shadow Volumetry", page_icon="🛢️")

st.title("🛢️ Oil Tank Volumetry AI")
st.markdown("Estimate internal oil volume using **Shadow Geometry** and **Solar Metadata**.")

# Sidebar for inputs
st.sidebar.header("Parameters")
elevation = st.sidebar.slider("Sun Elevation (Degrees)", 0, 90, 45)
uploaded_file = st.sidebar.file_ignore_case=True
uploaded_file = st.sidebar.file_uploader("Upload Tank Image", type=["png", "jpg", "jpeg"])

# Load Model
@st.cache_resource
def load_model():
    model = DeepShadowModel()
    # Ensure map_location='cpu' so it runs on Streamlit Cloud without a GPU
    model.load_state_dict(torch.load('best_deep_shadow_weights.pth', map_location='cpu'))
    model.eval()
    return model

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    st.image(img, caption="Uploaded Satellite/Simulated View", use_column_width=True)
    
    if st.button("Predict Volume"):
        model = load_model()
        
        # Preprocess
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)
        meta_tensor = torch.tensor([[elevation / 90.0]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(img_tensor, meta_tensor).item() * 100
        
        st.success(f"### Predicted Volume: {prediction:.2f}%")
        st.progress(prediction / 100.0)
