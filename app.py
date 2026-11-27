
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Project title
st.title("🌿 AI-Based Plant Disease Detection")
st.write("Upload a plant leaf image to detect possible diseases using a CNN model.")

# Load trained model and class names
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "plant_disease_model.keras"
    labels_path = Path(__file__).parent / "class_names.json"

    model = tf.keras.models.load_model(model_path)

    with open(labels_path, "r") as file:
        class_names = json.load(file)

    return model, class_names

# Disease information
disease_info = {
    "Pepper__bell___Bacterial_spot": (
        "Bacterial spot",
        "Small dark spots may appear on leaves.",
        "Caused by bacterial infection.",
        "Use disease-free seeds and avoid wetting leaves."
    ),
    "Pepper__bell___healthy": (
        "Healthy Pepper Leaf",
        "No visible disease symptoms.",
        "No disease detected.",
        "Continue proper watering and plant care."
    ),
    "Potato___Early_blight": (
        "Potato Early Blight",
        "Brown spots with circular ring patterns.",
        "Caused by Alternaria solani fungus.",
        "Remove infected leaves and avoid overhead watering."
    ),
    "Potato___Late_blight": (
        "Potato Late Blight",
        "Dark, water-soaked patches on leaves.",
        "Caused by Phytophthora infestans.",
        "Remove infected plants and maintain good airflow."
    ),
    "Potato___healthy": (
        "Healthy Potato Leaf",
        "No visible disease symptoms.",
        "No disease detected.",
        "Continue proper plant care."
    ),
    "Tomato_Bacterial_spot": (
        "Tomato Bacterial Spot",
        "Small dark spots on leaves and stems.",
        "Caused by bacterial infection.",
        "Use healthy seeds and remove infected leaves."
    ),
    "Tomato_Early_blight": (
        "Tomato Early Blight",
        "Brown spots with concentric rings.",
        "Caused by Alternaria fungus.",
        "Remove infected leaves and avoid wet foliage."
    ),
    "Tomato_Late_blight": (
        "Tomato Late Blight",
        "Dark water-soaked patches on leaves.",
        "Caused by Phytophthora infestans.",
        "Remove infected material and avoid wet foliage."
    ),
    "Tomato_Leaf_Mold": (
        "Tomato Leaf Mold",
        "Yellow patches on upper leaves and mold underneath.",
        "Caused by Passalora fulva fungus.",
        "Improve ventilation and reduce leaf moisture."
    ),
    "Tomato_Septoria_leaf_spot": (
        "Tomato Septoria Leaf Spot",
        "Small circular spots with dark borders.",
        "Caused by Septoria lycopersici fungus.",
        "Remove affected leaves and avoid overhead watering."
    ),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Tomato Spider Mites",
        "Tiny yellow spots and possible webbing on leaves.",
        "Caused by spider mite infestation.",
        "Inspect leaves and manage mites appropriately."
    ),
    "Tomato__Target_Spot": (
        "Tomato Target Spot",
        "Brown spots with target-like rings.",
        "Caused by Corynespora cassiicola fungus.",
        "Remove infected leaves and improve airflow."
    ),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Tomato Yellow Leaf Curl Virus",
        "Yellowing, curling and reduced leaf growth.",
        "A viral disease commonly spread by whiteflies.",
        "Control whiteflies and remove infected plants."
    ),
    "Tomato__Tomato_mosaic_virus": (
        "Tomato Mosaic Virus",
        "Mosaic patterns and leaf distortion.",
        "Caused by Tomato mosaic virus.",
        "Remove infected plants and clean gardening tools."
    ),
    "Tomato_healthy": (
        "Healthy Tomato Leaf",
        "No visible disease symptoms.",
        "No disease detected.",
        "Continue proper plant care."
    )
}

# Load model
try:
    model, class_names = load_model()
    st.success("CNN model loaded successfully!")
except Exception as error:
    st.error("Unable to load the model. Check that the model and class_names.json files are uploaded.")
    st.exception(error)
    st.stop()

# Image upload
uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf Image", width=350)

    if st.button("Detect Plant Disease"):

        with st.spinner("Analyzing leaf image..."):

            # Preprocess image
            image = image.resize((128, 128))
            image_array = tf.keras.utils.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Predict
            predictions = model.predict(image_array, verbose=0)

            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index]) * 100

            predicted_class = class_names[predicted_index]

            # Display prediction
            st.subheader("Prediction Result")
            st.write("**Detected Class:**", predicted_class.replace("_", " "))
            st.metric("Confidence", f"{confidence:.2f}%")

            # Display disease information
            if predicted_class in disease_info:
                name, symptoms, cause, prevention = disease_info[predicted_class]

                st.subheader(name)
                st.write("**Symptoms:**", symptoms)
                st.write("**Cause:**", cause)
                st.write("**Prevention:**", prevention)

            # Top 3 predictions
            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(predictions[0])[-3:][::-1]

            for index in top_indices:
                st.write(
                    f"{class_names[index].replace('_', ' ')}: "
                    f"{predictions[0][index] * 100:.2f}%"
                )

            st.info(
                "This is an AI-based educational prediction. "
                "Please consult an agricultural expert before taking "
                "plant treatment decisions."
            )
