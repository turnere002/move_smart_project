# app/app_streamlit.py
"""
Run:
streamlit run app_streamlit.py
Note: requires media processing code available and installed packages.
"""
import streamlit as st
import tempfile
import subprocess
import pandas as pd
import joblib
from pathlib import Path
import os

st.title("MoveSmart – Knee ROM quick demo")

uploaded = st.file_uploader("Upload side-view knee video (MP4)", type=["mp4"])
model_path = st.text_input("Path to saved model (.joblib)", value="../models/rf.joblib")
if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    st.write("Saved temp video:", tfile.name)
    # call process_videos.py on this single file (simplest approach)
    st.write("Processing video (this may take ~10-30s)...")
    out_csv = Path("temp_angles.csv")
    # Use subprocess to call the process_videos script (assumes python path correct)
    cmd = f"python ../src/process_videos.py --input_dir {Path(tfile.name).parent} --output_dir ./"
    st.write("NOTE: This demo assumes process_videos.py accepts a single-file run; adapt as needed.")
    st.warning("This demo is a skeleton. For reliable web demos adapt process_videos to accept single video path and return CSV.")
    # skip actual processing in skeleton; instead explain next steps
    st.info("After processing, run feature_extraction.py to get features and ml_models.py to predict.")
    st.stop()