from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import streamlit as st
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet


facenet = FaceNet(
    detector=MPFaceDetection(),
    onnx_model_path="models/faceNet.onnx",
    anchors="faces",
    force_cpu=True,
)


recognize = False

capture = False

if "count" not in st.session_state:
    st.session_state.count = 0

name = ""

st.header("Face Recognition Attendance System")


with st.sidebar:
    st.title("Menu: ")
    st.success("First turn on the webcam, then do any one of the following tasks.")

    name = st.text_input("Enter the Name of the Employee")
    if st.toggle("Capture & Store Image"):
        if name:
            capture = True
            st.write("Hi " + name + " your Image has been Captured")
            st.session_state.count += 1
    else:
        capture = False

    st.warning("If you want to use Face Recognition, first disable the Capture Image")

    if st.toggle("Start Face Recognition"):
        recognize = True
    else:
        recognize = False


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    if recognize:
        img = facenet(img)

    if capture:
        facenet.detect_save_faces(img, name=name)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="key",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)
