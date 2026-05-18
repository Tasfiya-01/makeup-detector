import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Makeup Product Detector",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #1a1a2e;}
    .stApp {background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);}
    h1 {color: #e94560; text-align: center; font-size: 3rem; margin-bottom: 0;}
    .subtitle {color: #a8a8b3; text-align: center; font-size: 1.1rem; margin-bottom: 2rem;}
    .detection-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .item-card {
        background: linear-gradient(90deg, rgba(233,69,96,0.2), rgba(15,52,96,0.2));
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        color: white;
        font-size: 1rem;
    }
    .stat-card {
        background: rgba(233,69,96,0.15);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(233,69,96,0.3);
    }
    .stat-number {color: #e94560; font-size: 2rem; font-weight: bold;}
    .stat-label {color: #a8a8b3; font-size: 0.9rem;}
    .stButton>button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-size: 1.1rem;
        width: 100%;
    }
    .upload-text {color: #a8a8b3;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>💄 Makeup Product Detector</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered makeup detection using YOLOv8</p>', unsafe_allow_html=True)

st.markdown("---")

model = YOLO('best.pt')

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a makeup image",
        type=["jpg", "jpeg", "png"],
        help="Upload any image containing makeup products"
    )
    
    confidence = st.slider("🎯 Confidence Level", 10, 90, 25, 5)
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### 🔍 Detection Result")
    
    if uploaded_file and st.button("✨ Detect Makeup Products"):
        with st.spinner("Analyzing image..."):
            image = Image.open(uploaded_file)
            results = model.predict(source=image, conf=confidence/100)
            result_image = results[0].plot()
            result_image = Image.fromarray(result_image)
            
            detections = []
            item_counts = {}
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = float(box.conf[0]) * 100
                detections.append((class_name, conf_score))
                item_counts[class_name] = item_counts.get(class_name, 0) + 1
            
            st.image(result_image, caption="Detection Result", use_column_width=True)
            
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{len(detections)}</div><div class="stat-label">Items Found</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{len(item_counts)}</div><div class="stat-label">Categories</div></div>', unsafe_allow_html=True)
            with c3:
                avg_conf = sum([c for _, c in detections]) / len(detections) if detections else 0
                st.markdown(f'<div class="stat-card"><div class="stat-number">{avg_conf:.0f}%</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)
            
            if detections:
                st.markdown("### 📋 Detected Items")
                for class_name, conf_score in detections:
                    st.markdown(f'<div class="item-card">💄 <b>{class_name}</b> — {conf_score:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.warning("No makeup items detected! Try lowering the confidence level.")

st.markdown("---")
st.markdown('<p style="text-align:center; color:#a8a8b3;">Powered by YOLOv8 | Makeup Product Detection AI</p>', unsafe_allow_html=True)
