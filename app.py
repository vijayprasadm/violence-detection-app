import streamlit as st
from predict import predict
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Violence Detection App",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM STYLES ----------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
    }
    .subtitle {
        font-size:20px;
        color: #444;
        text-align: center;
        margin-bottom: 30px;
    }
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #FF1C1C;
        color: #fff;
    }
    footer {
        text-align: center;
        color: grey;
        font-size: 14px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------- HEADER ----------------------
st.markdown('<p class="main-title">🚨 Violence Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a video to detect if it contains violent activity</p>', unsafe_allow_html=True)
st.write("---")

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("👉 Supported formats: MP4, AVI, MOV")
st.sidebar.info("👨‍💻 Developed by Vijay Prasad")

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    save_path = os.path.join("temp_video.mp4")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(save_path)  # preview uploaded video 🎥
    
    st.write("🔍 Analyzing video... please wait")
    result = predict(save_path)
    
    if result == "Fight":
        st.error("🚨 Prediction: **Fight Detected!** 🔴")
    else:
        st.success("✅ Prediction: **Non-fight video** 🟢")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    """
    <footer>
        © 2025 Violence Detection App | Built with ❤️ using Streamlit
    </footer>
    """, unsafe_allow_html=True
)
