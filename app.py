__author__ = "Marlon"
__Cop__ = "BrainiaC©"

import numpy as np
import cv2
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

st.title('Detector Númerico - BrainiaC©')

dire = os.path.join(os.path.dirname(__file__), 'model')
model = load_model('model')

st.write("""###### Author - Marlon Sousa""")
st.write("""###### [Blog](https://marlonsousa.medium.com)""")

st.markdown('''
Tente Desenhar um Número!
''')



canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=300,
    height=300,
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28))
    st.write(f"""# Resultado: {np.argmax(val[0])}""")

