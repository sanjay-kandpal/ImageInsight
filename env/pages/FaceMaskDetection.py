import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

def set_background(background_image):
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url("{background_image}") center;
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("./model/background.png")

FACE_MASK_MODEL_PATH = './model/face_mask_model.pt'

# Loading the Face Mask Detection Model
model = YOLO(FACE_MASK_MODEL_PATH)

threshold = 0.30

# Model Prediction Function
def model_prediction(img, threshold):
    counter = 0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = model.predict(img)[0]

    for result in pred.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            img = pred[0].plot()

            counter += 1

    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_wth_box, counter

# Image Resize Function
@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

app_mode = st.selectbox("Select App Mode: ", ["Upload Photo", "Take Photo"])

if app_mode == "Upload Photo":
    st.markdown("---------")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Parameters:")
    detection_confidence = st.slider("Min Detection Confidence:", min_value=0.0, max_value=1.0, value=0.5)
    st.markdown("---------")

    img_file = st.file_uploader("Upload Image:", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = np.array(Image.open(img_file))
    else:
        st.write("Please upload an image.")
        st.stop()

    st.text("Original Image:")
    st.image(image)

    if st.button("Apply Detection"):

        output_image, mask_counter = model_prediction(image, detection_confidence)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.title("Output Image ✅:")
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.image(output_image, use_column_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        _, col2, _ = st.columns([0.5, 1, 0.5])

        kpil_text = col2.markdown("0")
        kpil_text.write(
            f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>Detected Face Mask:</p>{mask_counter}</h1>",
            unsafe_allow_html=True)

elif app_mode == "Take Photo":
    st.markdown("---------")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Parameters:")
    detection_confidence = st.slider("Min Detection Confidence:", min_value=0.0, max_value=1.0, value=0.5)
    st.markdown("---------")

    img_file = st.camera_input("Take a Photo: ")

    if img_file is not None:
        image = np.array(img_file)
    else:
        st.write("Please take a photo.")
        st.stop()

    st.text("Original Image:")
    st.image(image)

    if st.button("Apply Detection"):

        output_image, mask_counter = model_prediction(image, detection_confidence)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.title("Output Image ✅:")
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.image(output_image, use_column_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        _, col2, _ = st.columns([0.5, 1, 0.5])

        kpil_text = col2.markdown("0")
        kpil_text.write(
            f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>Detected Face Mask:</p>{mask_counter}</h1>",
            unsafe_allow_html=True)
