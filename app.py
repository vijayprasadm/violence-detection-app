import streamlit as st
from predict import predict
import os

st.title("Violence Detection in Videos 🏒🔥")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    save_path = os.path.join("temp_video.mp4")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    
    result = predict(save_path)
    
    if result == "Fight":
        st.error("Prediction: Fight video 🔴")
    else:
        st.success("Prediction: Non-fight video 🟢")
