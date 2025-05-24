import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("second_modelcd.keras")

st.title("Cats vs Dogs")

uploaded_image = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Preprocess image
    img = Image.open(uploaded_image).convert('RGB')
    img = img.resize((180, 180))
    
    # sisp image
    st.image(img, caption="Uploaded Image", width=300)
    
    # predict the value
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0]) * 100
    class_name = "dog" if prediction[0][0] > 0.5 else "cat"
    confidence = confidence if class_name == "dog" else 100 - confidence
    
    st.metric("Prediction", f"{class_name.title()} ({confidence:.1f}% confidence)")
    if confidence > 95:
        st.success("Cool! the model did a great jon")
        st.balloons()
    elif confidence < 60:
        st.warning("The prediction was kinda poor")