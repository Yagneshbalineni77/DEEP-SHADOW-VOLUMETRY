
# Deep-Shadow Volumetry: Oil Tank Monitoring AI 🛢️

This project implements a Multi-Modal Deep Learning pipeline to estimate the volume of oil in storage tanks using satellite imagery and solar metadata.

### 🔗 Dataset
The model was trained on a custom synthetic dataset generated in Blender:
**[View Dataset on Kaggle](https://www.kaggle.com/datasets/yagneshchowdary007/oiltank)**

### 📊 Model Performance
- **Backbone:** EfficientNet-B0 (Transfer Learning)
- **Input:** 224x224 Grayscale Image + Solar Elevation Angle
- **Training:** 18 Epochs (Early Stopping triggered)
- **Mean Error:** ~1.6% to 2.0% on unseen test data

### 🛠️ Project Structure
- `best_deep_shadow_weights.pth`: The trained PyTorch model weights.
- `app.py`: Streamlit Dashboard for real-time inference.
- `notebook.ipynb`: The full training and validation pipeline.

### 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install torch torchvision streamlit pillow matplotlib`
3. Run the dashboard: `streamlit run app.py`
