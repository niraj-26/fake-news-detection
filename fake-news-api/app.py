import streamlit as st
import requests

st.title("Fake News Detection")

news = st.text_area("Enter News Text")

if st.button("Predict"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": news}
    )

    result = response.json()

    st.write("Prediction:", result["label"])
    st.write("Confidence:", result["confidence"])