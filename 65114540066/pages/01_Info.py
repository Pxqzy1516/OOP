import streamlit as st
from PIL import Image
from streamlit_extras.stoggle import stoggle


imcup = Image.open('cup.png')
st.set_page_config(
    page_title='CAFFEINE4TODAY',
    page_icon=(imcup),
    layout='centered')

st.title('Infographic about Caffeine🥤')
image = Image.open('caffeineperday.jpg')
st.image(image)

stoggle(
    "Something secret",
    'Have a coffee day ☕'
)
