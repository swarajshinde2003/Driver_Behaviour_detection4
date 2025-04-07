import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----- Streamlit Config -----
st.set_page_config(page_title="Driver Behavior Detector 🚗", page_icon="🚘", layout="wide")

@st.cache_resource
def load_cnn_model():
    return load_model("vggnet_best_model.h5")

model = load_cnn_model()
class_labels = ['Other', 'Safe', 'Talk', 'Text', 'Turn']

# ----- Custom CSS Styling -----
st.markdown("""
    <style>
        /* Fonts and layout */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
        }
        .main-title {
            font-size: 3.2em;
            font-weight: 800;
            background: linear-gradient(to right, #4A90E2, #50E3C2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2em;
        }
        .subtitle {
            font-size: 1.3em;
            color: #444;
            margin-bottom: 1.5em;
        }
        .prediction-box {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
            transition: 0.3s;
        }
        .prediction-box:hover {
            box-shadow: 0 12px 32px rgba(0,0,0,0.2);
        }
        .confidence {
            color: #F5A623;
            font-size: 1.3em;
            font-weight: 700;
        }
        .stDownloadButton button {
            background-color: #4A90E2;
            color: white;
            border-radius: 12px;
            padding: 0.6em 1.2em;
            transition: 0.3s;
        }
        .stDownloadButton button:hover {
            background-color: #357ABD;
        }
        .footer {
            text-align: center;
            margin-top: 4em;
            font-size: 0.85em;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# ----- Header -----
st.markdown('<div class="main-title"> Driver Behavior Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image or explore samples to detect driver behavior using deep learning.</div>', unsafe_allow_html=True)

# ----- Prediction Function -----
def predict_uploaded_image(img, model, class_labels, target_size=(240, 240)):
    img = img.convert("RGB")
    img_resized = img.resize(target_size)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index]
    return predicted_class, confidence, img_resized

# ----- Sidebar Navigation -----
st.sidebar.header("🔧 Navigation")
option = st.sidebar.radio("Select Mode:", ["📤 Upload Image", "🖼️ Try Sample Images"])

# ----- Upload Image -----
if option == "📤 Upload Image":
    st.subheader("📸 Upload a driver image for prediction")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        pred_class, confidence, display_img = predict_uploaded_image(img, model, class_labels)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🧠 Prediction: `{pred_class}`")
            st.markdown(f"<div class='confidence'>Confidence: {confidence:.2%}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ----- Sample Images -----
elif option == "🖼️ Try Sample Images":
    st.subheader("🧪 Select a category to preview & test with sample images")
    sample_dir = "sample_images"
    selected_class = st.selectbox("Choose a class to view samples", class_labels)
    class_path = os.path.join(sample_dir, selected_class)

    if os.path.exists(class_path):
        images = sorted(os.listdir(class_path))[:2]
        cols = st.columns(2)
        for i, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            with open(img_path, "rb") as file:
                cols[i].image(img_path, caption=f"{selected_class}", use_container_width=True)
                cols[i].download_button(
                    label="⬇️ Download Sample",
                    data=file,
                    file_name=img_name,
                    mime="image/jpeg",
                    key=f"download_{img_name}"
                )
    else:
        st.warning("⚠️ No sample images found in this category.")

# ----- Footer -----
st.markdown('<div class="footer">Made by Swaraj Shinde • Powered by VGGNet(CNN) •</div>', unsafe_allow_html=True)
