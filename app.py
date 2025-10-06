import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from collections import Counter
import pandas as pd

# -----------------------------
# 🔹 Judul & Deskripsi Aplikasi
# -----------------------------
st.set_page_config(page_title="Deteksi Kerusakan Dinding", page_icon="🏗️", layout="wide")
st.title("🏗️ Deteksi Jenis Kerusakan Dinding Menggunakan YOLOv8")

st.markdown("""
Aplikasi ini menggunakan model **YOLOv8** untuk mendeteksi berbagai jenis **keretakan dinding** dan **jamur**.

### Jenis kerusakan yang dapat dikenali:
- 🧱 `crack`
- 🔺 `diagonal_crack`
- 🩶 `hairline_crack`
- ➖ `horizontal_crack`
- ⚡ `through_crack`
- ⬆️ `vertical_crack`
- 🍃 `wall_mold`
""")

# -----------------------------
# 🔹 Load Model YOLOv8
# -----------------------------
MODEL_PATH = "best.pt"  # ganti jika model kamu di folder lain

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File model '{MODEL_PATH}' tidak ditemukan. Pastikan file tersebut ada di direktori aplikasi ini.")
    st.stop()

model = YOLO(MODEL_PATH)

# -----------------------------
# 🔹 Upload Gambar
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload gambar dinding (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Simpan gambar sementara
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Tampilkan gambar input
    image = Image.open(temp_file.name)
    st.image(image, caption="🖼️ Gambar Input", use_column_width=True)

    # -----------------------------
    # 🔹 Jalankan Prediksi YOLOv8
    # -----------------------------
    st.write("🔍 Sedang melakukan deteksi...")
    results = model.predict(source=temp_file.name, conf=0.25, save=False)
    r = results[0]
    boxes = r.boxes

    if len(boxes) == 0:
        st.success("✅ Tidak ditemukan kerusakan atau jamur pada dinding ini.")
    else:
        st.subheader("📋 Hasil Deteksi:")

        deteksi_data = []  # untuk tabel dan chart
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]

            deteksi_data.append({"Label": label, "Confidence": round(conf, 2)})

            # Tampilkan hasil + saran
            st.write(f"- **{label}** (confidence: {conf:.2f})")

            if label == "crack":
                st.info("🧱 Retakan umum terdeteksi. Lakukan pengecekan dan tambal area retak kecil.")
            elif label == "diagonal_crack":
                st.warning("🔺 Retakan diagonal terdeteksi. Bisa mengindikasikan pergeseran struktur atau pondasi.")
            elif label == "hairline_crack":
                st.info("🩶 Retakan rambut (hairline) terdeteksi. Biasanya akibat penyusutan plester.")
            elif label == "horizontal_crack":
                st.error("➖ Retakan horizontal terdeteksi. Berpotensi masalah struktural serius.")
            elif label == "through_crack":
                st.error("⚡ Retakan tembus (through crack) terdeteksi. Segera lakukan perbaikan permanen.")
            elif label == "vertical_crack":
                st.warning("⬆️ Retakan vertikal terdeteksi. Periksa tekanan struktural atau pergerakan tanah.")
            elif label == "wall_mold":
                st.info("🍃 Jamur dinding terdeteksi. Periksa kelembaban dan lakukan pembersihan.")

        # -----------------------------
        # 🔹 Simpan hasil deteksi
        # -----------------------------
        output_path = "output.jpg"
        r.save(filename=output_path)
        st.image(output_path, caption="📸 Hasil Deteksi dengan Bounding Box", use_column_width=True)

        # -----------------------------
        # 🔹 Tabel & Grafik Analisis
        # -----------------------------
        st.subheader("📊 Statistik Deteksi")
        df = pd.DataFrame(deteksi_data)
        st.dataframe(df, use_container_width=True)

        label_counts = Counter([d["Label"] for d in deteksi_data])
        st.bar_chart(label_counts)

        # Tombol untuk download hasil deteksi
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Unduh Gambar Hasil Deteksi",
                data=file,
                file_name="hasil_deteksi.jpg",
                mime="image/jpeg"
            )

    # Hapus file sementara setelah selesai
    os.remove(temp_file.name)
