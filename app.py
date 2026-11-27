import streamlit as st
import tensorflow as tf
import numpy as np
import json
import gdown
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Project title
st.title("🌿 AI-Based Plant Disease Detection")
st.write(
    "Upload a plant leaf image to detect possible diseases "
    "using a Convolutional Neural Network (CNN)."
)

# File paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "plant_disease_model.keras"
LABELS_PATH = BASE_DIR / "class_names.json"

# Google Drive model ID
MODEL_FILE_ID = "1MSwejLEL5tP9b1uQhaQYvvE6LUIWsNH0"

# Load trained model
@st.cache_resource
def load_model():

    # Download model if it is not available
    if not MODEL_PATH.exists():
        with st.spinner("Downloading trained CNN model..."):
            result = gdown.download(
                id=MODEL_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False
            )

        if result is None or not MODEL_PATH.exists():
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing permissions."
            )

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class names
    with open(LABELS_PATH, "r") as file:
        class_names = json.load(file)

    return model, class_names


# Disease information
disease_info = {

    "Pepper__bell___Bacterial_spot": (
        "Pepper Bacterial Spot",
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


# Load model safely
try:
    model, class_names = load_model()
    st.success("CNN model loaded successfully!")

except Exception as error:
    st.error("Unable to load the CNN model.")
    st.error(str(error))
    st.info(
        "Check that class_names.json is uploaded to GitHub "
        "and Google Drive sharing is set to Anyone with the link."
    )
    st.stop()


# Image upload section
st.subheader("Upload Plant Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        width=350
    )

    if st.button("Detect Plant Disease", type="primary"):

        with st.spinner("Analyzing the leaf image..."):

            # Resize image
            resized_image = image.resize((128, 128))

            # Convert image to array
            image_array = tf.keras.utils.img_to_array(
                resized_image
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Predict disease
            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[0][predicted_index] * 100
            )

        # Display result
        st.subheader("Prediction Result")

        st.write(
            "**Detected Class:**",
            predicted_class.replace("_", " ")
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Display disease information
        if predicted_class in disease_info:

            name, symptoms, cause, prevention = (
                disease_info[predicted_class]
            )

            st.subheader(name)

            st.write("**Symptoms:**", symptoms)

            st.write("**Cause:**", cause)

            st.write("**Prevention:**", prevention)

        # Display top 3 predictions
        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]

        for index in top_indices:

            st.write(
                f"**{class_names[index].replace('_', ' ')}** "
                f"- {predictions[0][index] * 100:.2f}%"
            )

        st.info(
            "This application is designed for educational purposes. "
            "AI predictions are not a substitute for professional "
            "agricultural diagnosis."
        )


# Footer
st.divider()

st.caption(
    "Developed using Python, TensorFlow, CNN and Streamlit."
)
