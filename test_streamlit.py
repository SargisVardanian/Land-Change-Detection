import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Test App")
img = np.zeros((500, 500, 3), dtype=np.uint8)
img[:, :] = [255, 0, 0]
selection = streamlit_image_coordinates(img, click_and_drag=True, key="test", width=250, height=250)
st.write(selection)
