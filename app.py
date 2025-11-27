import time

import streamlit as st
from ultralytics import YOLO
from PIL import Image

# =========================
# CONFIG APP
# =========================
st.set_page_config(
    page_title="Segmentasi Polip Kolonoskopi - YOLOv8-Seg",
    layout="wide"
)

st.title("🩺 Segmentasi Polip Kolonoskopi Real-Time – YOLOv8-Seg")
st.write("App berhasil dijalankan. Kalau kamu melihat teks ini, Streamlit sudah OK ✅")

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

# Di deployment, asumsikan best.pt ada di root repo
WEIGHTS_PATH = "best.pt"
st.sidebar.write("Model: `best.pt` (otomatis dibaca dari root project)")

img_size = st.sidebar.slider(
    "Image size (imgsz)",
    min_value=320,
    max_value=1024,
    value=640,
    step=32
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Pastikan file **best.pt** ada di root repo Streamlit (satu folder dengan app.py)."
)

# =========================
# LOAD MODEL (LAZY, CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO(WEIGHTS_PATH)

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
    # Tampilkan gambar original
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Gambar Original", use_column_width=True)

    # Load model hanya saat dibutuhkan
    with st.spinner("Memuat model YOLOv8-Seg... (pertama kali bisa agak lama)"):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            st.stop()

    st.subheader("2️⃣ Hasil Segmentasi YOLOv8-Seg")

    with st.spinner("Sedang melakukan segmentasi..."):
        t0 = time.time()
        results = model.predict(
            image,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False
        )
        dt = time.time() - t0

    result = results[0]

    # Visualisasi hasil (YOLO mengembalikan BGR numpy array)
    plotted = result.plot()              # BGR
    plotted_rgb = plotted[:, :, ::-1]    # ke RGB

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

    if result.masks is not None:
        num_polyp = len(result.masks.data)
        st.write(f"- Jumlah objek polip yang tersegmentasi: **{num_polyp}**")
else:
    st.info("Silakan upload gambar kolonoskopi terlebih dahulu.")

