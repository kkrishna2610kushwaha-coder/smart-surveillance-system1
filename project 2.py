
import os
import cv2
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Smart Surveillance System")
st.subheader(
    "Employee Monitoring and Indexing using Deep Learning"
)


# ==========================================
# 2. DATABASE CONFIGURATION
# ==========================================

DATABASE = "employee_records.csv"

COLUMNS = [
    "Date",
    "Time",
    "Person_Count",
    "Tracking_IDs",
    "Source"
]


def read_records():

    try:
        if os.path.exists(DATABASE):
            return pd.read_csv(DATABASE)

        return pd.DataFrame(columns=COLUMNS)

    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def save_records(records):

    if not records:
        return

    try:

        df = pd.DataFrame(records, columns=COLUMNS)

        file_exists = (
            os.path.exists(DATABASE)
            and os.path.getsize(DATABASE) > 0
        )

        df.to_csv(
            DATABASE,
            mode="a",
            header=not file_exists,
            index=False
        )

    except Exception as e:
        st.error(f"Database Error: {e}")


# ==========================================
# 3. LOAD YOLO11 MODEL
# ==========================================

@st.cache_resource
def load_ai_model():

    model = YOLO("yolo11n.pt")

    return model


try:

    model = load_ai_model()

    st.sidebar.success(
        "YOLO11 Model Loaded Successfully"
    )

except Exception as e:

    st.error(f"AI Model Loading Error: {e}")

    st.info(
        "Please check internet connection, "
        "model download and requirements.txt."
    )

    st.stop()


# ==========================================
# 4. SIDEBAR MENU
# ==========================================

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


# ==========================================
# 5. HOME DASHBOARD
# ==========================================

if menu == "Home":

    st.header("AI Surveillance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "YOLO11n")

    col2.metric("Technology", "Deep Learning")

    col3.metric("System", "Active")

    st.write(
        "Welcome to Smart Surveillance System."
    )

    st.info(
        "Upload an image or CCTV video "
        "to detect and track persons automatically."
    )

    st.markdown("### Project Features")

    st.write("1. AI-based Person Detection")

    st.write("2. CCTV Video Monitoring")

    st.write("3. Person Tracking using ByteTrack")

    st.write("4. Employee Monitoring Records")

    st.write("5. CSV Report Generation")

    st.write("6. Automatic AI-based Detection")


# ==========================================
# 6. IMAGE DETECTION MODULE
# ==========================================

elif menu == "Image Detection":

    st.header("🖼️ Person Detection")

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file is not None:

        try:

            image = np.array(
                Image.open(image_file).convert("RGB")
            )

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            if st.button("Start Detection"):

                with st.spinner("Detecting Persons..."):

                    results = model.predict(
                        source=image,
                        classes=[0],
                        conf=0.4,
                        verbose=False
                    )

                    result = results[0]

                    person_count = (
                        len(result.boxes)
                        if result.boxes is not None
                        else 0
                    )

                    output = result.plot()

                    output = cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB
                    )

                st.image(
                    output,
                    caption="Person Detection Result",
                    use_container_width=True
                )

                st.success(
                    f"Persons Detected: {person_count}"
                )

                record = [{
                    "Date": datetime.now().strftime(
                        "%Y-%m-%d"
                    ),
                    "Time": datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    "Person_Count": person_count,
                    "Tracking_IDs": "",
                    "Source": image_file.name
                }]

                save_records(record)

                st.success(
                    "Employee Record Saved Successfully!"
                )

        except Exception as e:

            st.error(
                f"Image Processing Error: {e}"
            )


# ==========================================
# 7. VIDEO MONITORING MODULE
# ==========================================

elif menu == "Video Monitoring":

    st.header("🎥 CCTV Video Monitoring")

    video_file = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file is not None:

        st.video(video_file)

        if st.button("Start Monitoring"):

            suffix = os.path.splitext(
                video_file.name
            )[1] or ".mp4"

            temp_path = None

            cap = None

            try:

                # Save uploaded video temporarily

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        video_file.getbuffer()
                    )

                    temp_path = temp_file.name

                # Open video

                cap = cv2.VideoCapture(temp_path)

                if not cap.isOpened():

                    st.error(
                        "Unable to open this video."
                    )

                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 25

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                frame_count = 0

                records = []

                progress = st.progress(0)

                video_display = st.empty()

                st.info(
                    "Video monitoring started..."
                )

                # Process video frames

                while cap.isOpened():

                    if frame_count >= 3000:

                        st.warning(
                            "Maximum 3000 frame limit reached."
                        )

                        break

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.track(
                        source=frame,
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    result = results[0]

                    detected = (
                        len(result.boxes)
                        if result.boxes is not None
                        else 0
                    )

                    ids = []

                    if (
                        result.boxes is not None
                        and result.boxes.id is not None
                    ):

                        ids = (
                            result.boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                    # Draw detection boxes

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

                    # Save records every second

                    if frame_count % max(
                        1, int(fps)
                    ) == 0:

                        records.append({

                            "Date": datetime.now().strftime(
                                "%Y-%m-%d"
                            ),

                            "Time": str(
                                round(frame_count / fps, 2)
                            ),

                            "Person_Count": detected,

                            "Tracking_IDs": str(ids),

                            "Source": video_file.name

                        })

                    frame_count += 1

                    # Update progress bar

                    if total_frames > 0:

                        progress.progress(
                            min(
                                frame_count / total_frames,
                                1.0
                            )
                        )

                # Save monitoring records

                if records:

                    save_records(records)

                    df = pd.DataFrame(
                        records,
                        columns=COLUMNS
                    )

                    st.success(
                        "Video Monitoring Completed Successfully!"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Total Frames Processed",
                        frame_count
                    )

                    col2.metric(
                        "Total Monitoring Records",
                        len(records)
                    )

                    st.subheader(
                        "Video Monitoring Report"
                    )

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    st.download_button(
                        label="Download Monitoring Report",
                        data=df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="surveillance_report.csv",
                        mime="text/csv"
                    )

                else:

                    st.warning(
                        "No video frames were processed."
                    )

            except Exception as e:

                st.error(
                    f"Video Processing Error: {e}"
                )

            finally:

                if cap is not None:
                    cap.release()

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# ==========================================
# 8. EMPLOYEE RECORDS MODULE
# ==========================================

elif menu == "Employee Records":

    st.header("📊 Employee Monitoring Records")

    data = read_records()

    st.dataframe(
        data,
        use_container_width=True
    )

    st.metric(
        "Total Records",
        len(data)
    )

    st.download_button(
        label="Download All Records",

        data=data.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="employee_records.csv",

        mime="text/csv"
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Surveillance System | Python | "
    "YOLO11 | OpenCV | Deep Learning"
)
