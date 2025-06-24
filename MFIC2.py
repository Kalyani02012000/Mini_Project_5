import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the best model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_fish_model.h5")
    return model

model = load_model()

# Define class labels (should match training classes)
class_labels = ['animal fish', 'animal fish bass', 'black_sea_sprat', 'gilt_head_bream',
                'hourse_mackerel', 'red_mullet', 'red_sea_bream', 'sea_bass', 'shrimp',
                'striped_red_mullet', 'trout']

# Set app title
st.title("🐟 Fish Species Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    st.write("⏳ Processing image...")
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize like training

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]

    # Display result
    st.success(f"✅ Predicted: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2%}**")

    # Show all class probabilities
    st.subheader("🔢 All Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob:.2%}")
