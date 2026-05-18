import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gdown
import os

if not os.path.exists('best.pt'):
    gdown.download(
        'https://drive.google.com/uc?id=1juFKcxrejHPvYu_tBWPlkx1uHnL73mKh',
        'best.pt', quiet=False
    )

st.set_page_config(
    page_title="Makeup Product Detector",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * {font-family: 'Poppins', sans-serif;}
    .stApp {
        background: linear-gradient(135deg, #f5f0ff 0%, #fff0f5 50%, #f0f5ff 100%);
    }
    .header-box {
        background: linear-gradient(135deg, #FF6B9D, #C44FD8, #7B6CF6);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(196,79,216,0.3);
    }
    .header-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .header-sub {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 10px;
    }
    .section-title {
        color: #C44FD8;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stat-box {
        background: linear-gradient(135deg, #FF6B9D22, #C44FD822);
        border: 2px solid #C44FD833;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    .stat-num {
        font-size: 2.2rem;
        font-weight: 700;
        color: #C44FD8;
        line-height: 1;
    }
    .stat-lab {
        font-size: 0.85rem;
        color: #888;
        margin-top: 5px;
    }
    .detection-item {
        background: linear-gradient(90deg, #fff0f9, #f5f0ff);
        border-left: 4px solid #C44FD8;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 8px 0;
    }
    .det-name {
        color: #333;
        font-weight: 600;
        font-size: 1rem;
    }
    .det-conf {
        color: #C44FD8;
        font-weight: 700;
        font-size: 1rem;
        float: right;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B9D, #C44FD8, #7B6CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 40px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 5px 20px rgba(196,79,216,0.4) !important;
    }
    .no-detection {
        background: #fff8f0;
        border: 2px dashed #ffaa44;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: #ff8844;
        font-size: 1.1rem;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.9rem;
        margin-top: 30px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-box">
        <div class="header-title">💄 Makeup Product Detector</div>
        <div class="header-sub">Upload any makeup image and let AI identify the products instantly!</div>
    </div>
""", unsafe_allow_html=True)

model = YOLO('best.pt')

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📤 Upload Your Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png"],
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Detection Sensitivity</div>', unsafe_allow_html=True)
    confidence = st.slider("", 10, 90, 25, 5, format="%d%%")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded image", use_column_width=True)

with col2:
    st.markdown('<div class="section-title">🔍 Detection Results</div>', unsafe_allow_html=True)

    if uploaded_file:
        if st.button("✨ Detect Makeup Products"):
            with st.spinner("AI is analyzing your image..."):
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

                st.image(result_image, use_column_width=True)

                if detections:
                    avg_conf = sum([c for _, c in detections]) / len(detections)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(detections)}</div><div class="stat-lab">Items Found</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(item_counts)}</div><div class="stat-lab">Categories</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_conf:.0f}%</div><div class="stat-lab">Avg Confidence</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📋 Detected Products</div>', unsafe_allow_html=True)
                    for class_name, conf_score in detections:
                        st.markdown(f'<div class="detection-item"><span class="det-name">💄 {class_name}</span><span class="det-conf">{conf_score:.1f}%</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-detection">😕 No makeup products detected!<br><small>Try lowering the sensitivity slider</small></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center; color:#ccc; padding:80px 20px;">
                <div style="font-size:4rem;">💄</div>
                <div style="font-size:1.1rem; margin-top:15px;">Upload an image to get started!</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        Powered by YOLOv8 AI Model &nbsp;|&nbsp; 31 Makeup Categories &nbsp;|&nbsp; Real-time Detection
    </div>
""", unsafe_allow_html=True)
