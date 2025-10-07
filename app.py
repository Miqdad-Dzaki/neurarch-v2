import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from collections import Counter
import pandas as pd

# -----------------------------
# 🏗️ APP CONFIG
# -----------------------------
st.set_page_config(page_title="Wall Damage Detection", page_icon="🏗️", layout="wide")
st.title("🏗️ Wall Crack and Mold Detection using YOLOv8")

st.markdown("""
This app uses a **YOLOv8 model** to detect various **wall cracks** and **black mold** types.
Each detection result includes a brief explanation and the recommended repair method.
""")

# -----------------------------
# 🔹 LOAD YOLO MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file `best.pt` not found. Please upload it to the app directory.")
    st.stop()

model = load_model()

# -----------------------------
# 🔹 IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload wall image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Display the uploaded image
    image = Image.open(temp_file.name)
    st.image(image, caption="🖼️ Input Image", use_column_width=True)

    # -----------------------------
    # 🔹 YOLO PREDICTION
    # -----------------------------
    st.write("🔍 Detecting damage...")
    results = model.predict(source=temp_file.name, conf=0.25, save=False)
    r = results[0]
    boxes = r.boxes

    if len(boxes) == 0:
        st.success("✅ No visible damage or mold detected on this wall.")
    else:
        st.subheader("📋 Detection Results")

        deteksi_data = []  # For dataframe and charts

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]
            deteksi_data.append({"Label": label, "Confidence": round(conf, 2)})

            st.write(f"- **{label}** (confidence: {conf:.2f})")

            # -----------------------------
            # 🔹 CRACK & MOLD SOLUTIONS
            # -----------------------------
            if label == "hairline_crack":
                st.info("🩶 Hairline Crack — very thin, like a hair strand. Usually harmless but can allow moisture seepage. \
**Solution:** Fill with elastic filler or acrylic sealant.")
            elif label == "crack":
                st.info("🧱 Small Crack — less than 3 mm wide. Caused by shrinkage or minor movement. \
**Solution:** Monitor; if spreading, use acrylic sealant.")
            elif label == "vertical_crack":
                st.warning("⬆️ Vertical Crack — straight line from top to bottom, may let water seep through. \
**Solution:** Evaluate the foundation if >5 mm; repair with polymer-modified mortar.")
            elif label == "diagonal_crack":
                st.error("🔺 Diagonal Crack — slanted line indicating uneven foundation settlement. \
**Solution:** Investigate foundation stability and repair with structural mortar.")
            elif label == "horizontal_crack":
                st.error("➖ Horizontal Crack — runs across the wall; highly dangerous. \
**Solution:** Indicates lateral pressure or soil shift — requires immediate structural repair.")
            elif label == "through_crack":
                st.error("⚡ Through Crack — extends across multiple bricks; serious structural damage. \
**Solution:** Use epoxy injection or contact a structural engineer immediately.")
            elif label == "wall_mold":
                st.warning("🍃 Black Mold Detected — caused by high humidity or water leakage. \
**Solution:** Clean using **baking soda**, **bleach (1:10)**, or **white vinegar**. \
Ensure proper ventilation and fix leaks to prevent recurrence.")

        # -----------------------------
        # 🔹 SAVE & SHOW RESULT IMAGE
        # -----------------------------
        output_path = "output.jpg"
        r.save(filename=output_path)
        st.image(output_path, caption="📸 Detection with Bounding Boxes", use_column_width=True)

        # -----------------------------
        # 🔹 STATISTICS
        # -----------------------------
        st.subheader("📊 Detection Summary")
        df = pd.DataFrame(deteksi_data)
        st.dataframe(df, use_container_width=True)

        label_counts = Counter([d["Label"] for d in deteksi_data])
        st.bar_chart(label_counts)

        # -----------------------------
        # 🔹 DOWNLOAD OPTION
        # -----------------------------
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Detection Image",
                data=file,
                file_name="detected_wall_damage.jpg",
                mime="image/jpeg"
            )

    # Clean up temporary file
    os.remove(temp_file.name)
