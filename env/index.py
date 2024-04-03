import streamlit as st
from PIL import Image


import numpy as np
from streamlit_extras.buy_me_a_coffee import button
st.title('Image Insight',)
   
col1, col2 = st.columns(2)
with col1:
    st.image('output.png', use_column_width=True ,width=300)
with col2:
    st.image('frames.png',use_column_width=True,width=300)

st.subheader('A simple Application that shows OpenCv Power various use Cases , just by using Open Cv Library.You can choose the options'
             + 'from the left. I have implemented only few to show the power of OpenCv How it works in Streamlit. '+
             'You are free to add stuff to this app.')


button(username="Sanjay Kandpal", floating=True, width=221)

if st.button("Read More"):
     st.markdown("<meta http-equiv='refresh' content='0;URL=http://localhost:8501/ImageEditing'>", unsafe_allow_html=True)