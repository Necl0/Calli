import streamlit as st
from main import *

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def load_image():
    uploaded_file = st.file_uploader(label='Upload an image:')

    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        return image_data


def main():
    st.title('Bonsai')
    pred = model(load_image())
    st.write(f"Prediction: {pred}")


main()


