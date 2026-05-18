import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Makeup Product Detector")
st.write("Upload an image to detect makeup products!")

model = YOLO('best.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Makeup"):
        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=0.25)
            result_image = results[0].plot()
            result_image = Image.fromarray(result_image)
            
            st.image(result_image, caption="Detection Result", use_column_width=True)
            
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0]) * 100
                detections.append(f"{class_name}: {confidence:.1f}%")
            
            if detections:
                st.success(f"Total items detected: {len(detections)}")
                st.write("Detected Items:")
                for item in detections:
                    st.write(f"✅ {item}")
            else:
                st.warning("No makeup items detected!")
