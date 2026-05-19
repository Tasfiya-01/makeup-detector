import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gdown
import os
import io
import plotly.graph_objects as go

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
        border-radius: 25px;
        padding: 45px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(196,79,216,0.4);
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0% {box-shadow: 0 10px 40px rgba(196,79,216,0.4);}
        50% {box-shadow: 0 10px 60px rgba(196,79,216,0.7);}
        100% {box-shadow: 0 10px 40px rgba(196,79,216,0.4);}
    }
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .header-sub {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 10px;
    }
    .glass-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(196,79,216,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        margin-bottom: 20px;
    }
    .section-title {
        color: #C44FD8;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stat-box {
        background: linear-gradient(135deg, #FF6B9D22, #C44FD822);
        border: 2px solid #C44FD844;
        border-radius: 15px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s;
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 700;
        color: #C44FD8;
        line-height: 1;
    }
    .stat-lab {
        font-size: 0.8rem;
        color: #888;
        margin-top: 5px;
    }
    .det-high {
        background: linear-gradient(90deg, #f0fff4, #e6ffed);
        border-left: 4px solid #28a745;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 6px 0;
    }
    .det-mid {
        background: linear-gradient(90deg, #fffdf0, #fff9e6);
        border-left: 4px solid #ffc107;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 6px 0;
    }
    .det-low {
        background: linear-gradient(90deg, #fff5f5, #ffe6e6);
        border-left: 4px solid #dc3545;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 6px 0;
    }
    .det-name {color: #333; font-weight: 600; font-size: 0.95rem;}
    .det-conf {font-weight: 700; font-size: 0.95rem; float: right;}
    .conf-high {color: #28a745;}
    .conf-mid {color: #ffc107;}
    .conf-low {color: #dc3545;}
    .history-item {
        background: rgba(196,79,216,0.08);
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        font-size: 0.9rem;
        color: #555;
    }
    .about-card {
        background: linear-gradient(135deg, #C44FD811, #7B6CF611);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #C44FD833;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B9D, #C44FD8, #7B6CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 5px 20px rgba(196,79,216,0.4) !important;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 30px;
        padding: 20px;
        border-top: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'total_images' not in st.session_state:
    st.session_state.total_images = 0

st.markdown("""
    <div class="header-box">
        <div class="header-title">💄 Makeup Product Detector</div>
        <div class="header-sub">✨ Upload any makeup image and let AI identify the products instantly!</div>
    </div>
""", unsafe_allow_html=True)

model = YOLO('best.pt')

tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 History", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📤 Upload Your Image</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Detection Sensitivity</div>', unsafe_allow_html=True)
        confidence = st.slider("", 10, 90, 25, 5, format="%d%%")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_files:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🖼️ Uploaded Images</div>', unsafe_allow_html=True)
            for uf in uploaded_files:
                img = Image.open(uf)
                st.image(img, caption=uf.name, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Detection Results</div>', unsafe_allow_html=True)

        if uploaded_files:
            if st.button("✨ Detect Makeup Products"):
                all_detections = []
                for uf in uploaded_files:
                    image = Image.open(uf)
                    with st.spinner(f"Analyzing {uf.name}..."):
                        results = model.predict(source=image, conf=confidence/100)
                        result_image = results[0].plot()
                        result_image_pil = Image.fromarray(result_image)

                        detections = []
                        item_counts = {}
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            conf_score = float(box.conf[0]) * 100
                            detections.append((class_name, conf_score))
                            item_counts[class_name] = item_counts.get(class_name, 0) + 1
                            all_detections.append(class_name)

                        st.markdown(f"**{uf.name}**")
                        st.image(result_image_pil, use_column_width=True)

                        buf = io.BytesIO()
                        result_image_pil.save(buf, format='PNG')
                        st.download_button(
                            label="⬇️ Download Result",
                            data=buf.getvalue(),
                            file_name=f"detected_{uf.name}",
                            mime="image/png"
                        )

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
                                if conf_score >= 80:
                                    div_class = "det-high"
                                    conf_class = "conf-high"
                                    emoji = "🟢"
                                elif conf_score >= 50:
                                    div_class = "det-mid"
                                    conf_class = "conf-mid"
                                    emoji = "🟡"
                                else:
                                    div_class = "det-low"
                                    conf_class = "conf-low"
                                    emoji = "🔴"
                                st.markdown(f'<div class="{div_class}"><span class="det-name">{emoji} {class_name}</span><span class="det-conf {conf_class}">{conf_score:.1f}%</span></div>', unsafe_allow_html=True)

                            if len(item_counts) > 1:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown('<div class="section-title">📊 Detection Chart</div>', unsafe_allow_html=True)
                                fig = go.Figure(go.Bar(
                                    x=list(item_counts.keys()),
                                    y=list(item_counts.values()),
                                    marker_color='#C44FD8',
                                    marker_line_color='#7B6CF6',
                                    marker_line_width=1.5,
                                ))
                                fig.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font_color='#555',
                                    height=250,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No makeup products detected! Try lowering sensitivity.")

                st.session_state.total_images += len(uploaded_files)
                st.session_state.total_detections += len(all_detections)
                if all_detections:
                    st.session_state.history.append({
                        'images': len(uploaded_files),
                        'detections': len(all_detections),
                        'items': list(set(all_detections))
                    })
        else:
            st.markdown("""
                <div style="text-align:center; color:#ccc; padding:80px 20px;">
                    <div style="font-size:4rem;">💄</div>
                    <div style="font-size:1.1rem; margin-top:15px; color:#bbb;">Upload an image to get started!</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Session Statistics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.total_images}</div><div class="stat-lab">Images Analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.total_detections}</div><div class="stat-lab">Total Detections</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕓 Detection History</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            st.markdown(f'<div class="history-item">🔍 Scan #{len(st.session_state.history)-i} — {h["images"]} image(s) — {h["detections"]} detections — Items: {", ".join(h["items"])}</div>', unsafe_allow_html=True)
    else:
        st.info("No history yet! Go to Detect tab and analyze some images.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 About This App</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="about-card">
           <b>Dataset:</b> 1,500 training images, 413 validation images, 207 test images<br>
            <b>Validation mAP50:</b> 46.9% | <b>Test mAP50:</b> 48.9%<br>
            <b>Categories:</b> 31 makeup product classes
        </div>
        <div class="about-card">
            <b>💄 Detectable Products:</b><br>
            Lipstick, Lip Gloss, Lip Liner, Lip Balm, Blush, Bronzer, Highlighter,
            Foundation, BB Cream, Concealer, Powder, Primer, Serum, Lotion,
            Eye Liner, Eye Shade, Mascara, Kajal, Tint, Brush, Beauty Blender,
            Eyelash Curler, Nail Polish, Nail Polish Remover, Gel, Makeup,
            Perfume, Setting Spray, Sun Block
        </div>
        <div class="about-card">
            <b>🟢 High Confidence (80%+):</b> Very accurate detection<br>
            <b>🟡 Medium Confidence (50-80%):</b> Likely correct<br>
            <b>🔴 Low Confidence (below 50%):</b> Uncertain detection
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        💄 Makeup Product Detector &nbsp;|&nbsp; Powered by YOLOv8 &nbsp;|&nbsp; 31 Categories &nbsp;|&nbsp; Real-time AI Detection
    </div>
""", unsafe_allow_html=True)
