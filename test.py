import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import imutils

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        video = frame.to_ndarray(format="bgr24")
        video = imutils.resize(video, width=400)
        (h, w) = video.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(video, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(video, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(video, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return video

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Dang Huu Giau
            MSSV: 20110636   
            Email : 20110636@student.hcmute.edu.vn""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)


    else:
        pass


if __name__ == "__main__":
    main()