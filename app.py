import time
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# CONFIG APP
# =========================
st.set_page_config(
    page_title="Segmentasi Polip Kolonoskopi - YOLOv8-Seg",
    layout="wide"
)

st.title("🩺 Segmentasi Polip Kolonoskopi Real-Time – YOLOv8-Seg")

st.markdown(
    """
Aplikasi ini menggunakan **YOLOv8-Seg** untuk melakukan segmentasi polip pada citra kolonoskopi.  
Upload gambar, lalu model akan memberikan hasil segmentasi beserta estimasi **waktu pemrosesan** dan **FPS**.
"""
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan")

default_weights = "best.pt"  # ganti kalau file kamu beda nama/path
weights_path = st.sidebar.text_input(
    "Path model YOLOv8-Seg (.pt)",
    value=default_weights,
    help="Letakkan file best.pt di folder yang sama dengan app.py atau tuliskan path lengkapnya."
)

img_size = st.sidebar.slider(
    "Image size (imgsz)",
    min_value=320,
    max_value=1024,
    value=640,
    step=32
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Pastikan file **model** sudah ada di path yang benar sebelum menjalankan prediksi."
)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model(path: str):
    model = YOLO(path)
    return model

model = None
model_load_error = None
if weights_path:
    try:
        model = load_model(weights_path)
    except Exception as e:
        model_load_error = str(e)

# =========================
# MAIN CONTENT
# =========================
st.subheader("1️⃣ Upload Gambar Kolonoskopi")

uploaded_file = st.file_uploader(
    "Pilih file gambar (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Gambar Original", use_column_width=True)

    if model_load_error:
        st.error(f"Gagal load model: {model_load_error}")
    elif model is None:
        st.warning("Model belum dimuat. Cek kembali path model di sidebar.")
    else:
        st.subheader("2️⃣ Hasil Segmentasi YOLOv8-Seg")

        with st.spinner("Sedang melakukan segmentasi..."):
            # Konversi ke format yang bisa diterima YOLO langsung (PIL ok)
            t0 = time.time()
            results = model.predict(
                image,
                imgsz=img_size,
                verbose=False
            )
            dt = time.time() - t0

        result = results[0]

        # Visualisasi hasil (YOLO sudah mengembalikan numpy array BGR)
        plotted = result.plot()              # BGR
        plotted_rgb = plotted[:, :, ::-1]    # konversi ke RGB

        with col2:
            st.image(plotted_rgb, caption="Hasil Segmentasi", use_column_width=True)

        # =========================
        # METRIK WAKTU & FPS
        # =========================
        ms_per_frame = dt * 1000.0
        fps = 1.0 / dt if dt > 0 else 0.0

        st.markdown("### 3️⃣ Metrik Real-Time (Per Gambar)")
        st.write(f"- Waktu pemrosesan (inference time per frame): **{ms_per_frame:.2f} ms/frame**")
        st.write(f"- Perkiraan FPS: **{fps:.2f} frame/detik**")

        # Opsional: info jumlah mask (jumlah polip terdeteksi)
        if result.masks is not None:
            num_polyp = len(result.masks.data)
            st.write(f"- Jumlah objek polip yang tersegmentasi: **{num_polyp}**")
else:
    st.info("Silakan upload gambar kolonoskopi terlebih dahulu.")

