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
Upload **satu atau beberapa gambar**, lalu model akan memberikan hasil segmentasi beserta estimasi **waktu pemrosesan** dan **FPS**.
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

uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa file gambar (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if model_load_error:
        st.error(f"Gagal load model: {model_load_error}")
    elif model is None:
        st.warning("Model belum dimuat. Cek kembali path model di sidebar.")
    else:
        # List untuk menyimpan metrik semua gambar
        all_ms = []
        all_fps = []

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            st.markdown(f"---\n### 🖼️ Gambar #{idx}")

            col1, col2 = st.columns(2)

            # Baca gambar
            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(image, caption=f"Gambar Original #{idx}", use_column_width=True)

            with st.spinner(f"Sedang melakukan segmentasi untuk gambar #{idx}..."):
                t0 = time.time()
                results = model.predict(
                    image,
                    imgsz=img_size,
                    verbose=False
                )
                dt = time.time() - t0

            result = results[0]

            # Visualisasi hasil (YOLO mengembalikan numpy array BGR)
            plotted = result.plot()              # BGR
            plotted_rgb = plotted[:, :, ::-1]    # ke RGB

            with col2:
                st.image(plotted_rgb, caption=f"Hasil Segmentasi #{idx}", use_column_width=True)

            # Metrik per gambar
            ms_per_frame = dt * 1000.0
            fps = 1.0 / dt if dt > 0 else 0.0

            all_ms.append(ms_per_frame)
            all_fps.append(fps)

            st.write(f"- Waktu pemrosesan (inference time per frame): **{ms_per_frame:.2f} ms/frame**")
            st.write(f"- Perkiraan FPS: **{fps:.2f} frame/detik**")

            if result.masks is not None:
                num_polyp = len(result.masks.data)
                st.write(f"- Jumlah objek polip yang tersegmentasi: **{num_polyp}**")

        # =========================
        # RINGKASAN RATA-RATA & MEDIAN
        # =========================
        if all_fps:
            avg_ms = float(np.mean(all_ms))
            med_ms = float(np.median(all_ms))
            avg_fps = float(np.mean(all_fps))
            med_fps = float(np.median(all_fps))

            st.markdown("---")
            st.markdown("## 📊 Rangkuman Metrik Semua Gambar")
            st.write(f"- Rata-rata waktu pemrosesan: **{avg_ms:.2f} ms/frame**")
            st.write(f"- Median waktu pemrosesan: **{med_ms:.2f} ms/frame**")
            st.write(f"- Rata-rata FPS: **{avg_fps:.2f} frame/detik**")
            st.write(f"- Median FPS: **{med_fps:.2f} frame/detik**")
else:
    st.info("Silakan upload satu atau beberapa gambar kolonoskopi terlebih dahulu.")
