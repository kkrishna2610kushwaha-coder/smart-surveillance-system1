
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from datetime import datetime
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Smart Surveillance System")
st.subheader("Employee Monitoring and Indexing using Deep Learning")


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_ai_model():
    return YOLO("yolo11n.pt")


try:
    model = load_ai_model()
    st.sidebar.success("AI Model Loaded Successfully")
except Exception as e:
    st.error(f"AI Model Loading Error: {e}")
    st.stop()


# ---------------- DATABASE ----------------

DATABASE = "employee_records.csv"

if not os.path.exists(DATABASE):
    df = pd.DataFrame(columns=[
        "Date",
        "Time",
        "Person_Count",
        "Tracking_IDs",
        "Source"
    ])
    df.to_csv(DATABASE, index=False)


def save_records(records):
    if records:
        df = pd.DataFrame(records)
        df.to_csv(
            DATABASE,
            mode="a",
            header=False,
            index=False
        )


# ---------------- SIDEBAR ----------------

st.sidebar.title("MAIN MENU")

menu = st.sidebar.selectbox(
    "Choose Module",
    [
        "Home",
        "Image Detection",
        "Video Monitoring",
        "Employee Records"
    ]
)


# ---------------- HOME MODULE ----------------

if menu == "Home":

    st.header("AI Surveillance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "YOLO11")
    col2.metric("Technology", "Deep Learning")
    col3.metric("System", "Active")

    st.write(
        "Welcome to Smart Surveillance System."
    )

    st.info(
        "Upload an image or CCTV video "
        "to detect and track people."
    )


# ---------------- IMAGE DETECTION ----------------

elif menu == "Image Detection":

    st.header("🖼️ Person Detection")

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:

        image = np.array(
            Image.open(image_file).convert("RGB")
        )

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

        if st.button("Start Detection"):

            with st.spinner("Detecting persons..."):

                results = model.predict(
                    image,
                    classes=[0],
                    conf=0.4,
                    verbose=False
                )

                result = results[0]
                count = len(result.boxes)

                output = result.plot()

                output = cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                )

            st.image(
                output,
                caption="Detection Result",
                use_container_width=True
            )

            st.success(
                f"Persons Detected: {count}"
            )

            record = [{
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Person_Count": count,
                "Tracking_IDs": "",
                "Source": image_file.name
            }]

            save_records(record)

            st.success("Employee record saved!")


# ---------------- VIDEO MONITORING ----------------

elif menu == "Video Monitoring":

    st.header("🎥 CCTV Video Monitoring")

    video_file = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:

        st.video(video_file)

        if st.button("Start Monitoring"):

            os.makedirs("outputs", exist_ok=True)

            video_path = "outputs/input.mp4"

            with open(video_path, "wb") as file:
                file.write(video_file.getbuffer())

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                st.error("Unable to open this video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            total = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            frame_count = 0
            records = []

            progress = st.progress(0)
            video_display = st.empty()

            try:

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.track(
                        frame,
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    result = results[0]

                    detected = len(result.boxes)

                    ids = []

                    if result.boxes.id is not None:

                        ids = (
                            result.boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                    output = result.plot()

                    output = cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB
                    )

                    video_display.image(
                        output,
                        caption="Live Person Tracking",
                        use_container_width=True
                    )

                    if frame_count % max(1, int(fps)) == 0:

                        records.append({
                            "Date": datetime.now().strftime("%Y-%m-%d"),
                            "Time": str(round(frame_count / fps, 2)),
                            "Person_Count": detected,
                            "Tracking_IDs": str(ids),
                            "Source": video_file.name
                        })

                    frame_count += 1

                    if total > 0:

                        progress.progress(
                            min(frame_count / total, 1.0)
                        )

            finally:
                cap.release()

            if records:

                save_records(records)

                df = pd.DataFrame(records)

                st.success(
                    "Video Monitoring Completed!"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.download_button(
                    "Download Monitoring Report",
                    df.to_csv(index=False),
                    file_name="surveillance_report.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No video frames were processed."
                )


# ---------------- EMPLOYEE RECORDS ----------------

elif menu == "Employee Records":

    st.header("📊 Employee Monitoring Records")

    data = pd.read_csv(DATABASE)

    st.dataframe(
        data,
        use_container_width=True
    )

    st.metric(
        "Total Records",
        len(data)
    )

    st.download_button(
        "Download All Records",
        data.to_csv(index=False),
        file_name="employee_records.csv",
        mime="text/csv"
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Smart Surveillance System | Python | YOLO11 | "
    "OpenCV | Deep Learning"
)
