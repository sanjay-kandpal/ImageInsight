import streamlit as st
from PIL import Image
import cv2
import numpy as np

def enhance_image(image):
    # Sharpening the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Finding contours
    imgray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = cv2.drawContours(sharpened, contours, -1, (0,255,0), 3)
    
    return sharpened, img_with_contours

st.header('Thresholding, Edge Detection and Contours')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_array = np.array(image)
    
    
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    x = st.slider('Change Threshold value', min_value=50, max_value=255)
    ret, thresh1 = cv2.threshold(image_gray, x, 255, cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True, clamp=True)

    st.text('Bar Chart Of the image')
    histr = cv2.calcHist([image_gray], [0], None, [256], [0,256])
    st.bar_chart(histr)

    st.text('Press the button below to view canny Edge Detection Technique')
    if st.button('Canny Edge Detector'):
        edges = cv2.Canny(image_array, 50, 300)
        st.image(edges, use_column_width=True, clamp=True)

    y = st.slider('Change Value to increase or decrease contours', min_value=50, max_value=255)

    if st.button('Contours'):
        sharpened_im, im_with_contours = enhance_image(image_array)
        st.image(sharpened_im, use_column_width=True, clamp=True)
        st.image(im_with_contours, use_column_width=True, clamp=True)
