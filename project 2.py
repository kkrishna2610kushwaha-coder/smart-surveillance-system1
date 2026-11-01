import cv2
import pandas as pd
import streamlit as st
import os
from datetime import datetime
from ultralytics import YOLO

# PROJECT CONFIGURATION

st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="📹",
    layout="wide"
)

st.title("Smart Surveillance System")
st.subheader("Employee Monitoring and Indexing using Deep Learning")

# LOAD AI MODEL

@st.cache_resource
def load_ai_model():
    return YOLO("yolo11n.pt")

model = load_ai_model()

# CREATE DATABASE

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

# SIDEBAR

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

# HOME MODULE

if menu == "Home":

    st.header("AI Surveillance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "YOLO11")
    col2.metric("Technology", "Deep Learning")
    col3.metric("System", "Active")

    st.write("Welcome to Smart Surveillance System.")

    st.info(
        "Upload an image or video to detect and track people."
    )

# IMAGE DETECTION MODULE

elif menu == "Image Detection":

    st.header("Person Detection")

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if image_file:

        import numpy as np

        image = np.array(
            __import__("PIL").Image.open(image_file).convert("RGB")
        )

        st.image(image, caption="Original Image")

        if st.button("Start Detection"):

            results = model.predict(
                image,
                classes=[0],
                conf=0.4
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
                caption="Detection Result"
            )

            st.success(f"Persons Detected: {count}")

            record = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Person_Count": count,
                "Tracking_IDs": "",
                "Source": image_file.name
            }

            pd.DataFrame([record]).to_csv(
                DATABASE,
                mode="a",
                header=False,
                index=False
            )

# VIDEO MONITORING MODULE

elif menu == "Video Monitoring":

    st.header("Live Video Analysis")

    video_file = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:

        if st.button("Start Monitoring"):

            os.makedirs("outputs", exist_ok=True)

            video_path = "outputs/input.mp4"

            with open(video_path, "wb") as file:
                file.write(video_file.getbuffer())

            cap = cv2.VideoCapture(video_path)

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
                    ids = result.boxes.id.int().cpu().tolist()

                output = result.plot()

                video_display.image(
                    cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB
                    )
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

            cap.release()

            if records:

                df = pd.DataFrame(records)

                df.to_csv(
                    DATABASE,
                    mode="a",
                    header=False,
                    index=False
                )

                st.success("Monitoring Completed!")

                st.dataframe(df)

                st.download_button(
                    "Download Report",
                    df.to_csv(index=False),
                    file_name="surveillance_report.csv",
                    mime="text/csv"
                )

# EMPLOYEE RECORDS MODULE

elif menu == "Employee Records":

    st.header("Monitoring Records")

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

st.divider()

st.caption(
    "Smart Surveillance System | Python | YOLO | "
    "OpenCV | Deep Learning"
)