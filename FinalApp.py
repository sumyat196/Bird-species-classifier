import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Lambda

# Define the custom function if needed, replace this with the actual operation if known
def tf_op_lambda(x):
    # Replace with the actual function or operation used in your model
    return x

# Load the model with custom objects
model_path = r'C:\Users\Admin\Downloads\BetterModel.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'TFOpLambda': Lambda(tf_op_lambda)})

# Set up the page configuration
st.set_page_config(page_title="Bird Species Classifier", layout="wide")

# Display the background image using a container with adjusted styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.goodfon.com/original/3840x2160/d/fb/belogolovyi-orlan-ptitsa-orel-chernyi-fon-portret-vzgliad-kh.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Bird Species Classifier")

uploaded_file = st.file_uploader("Choose a bird image...", type=["jpg", "jpeg", "png"])

# Map dictionary for known bird species
map_dict = {
    0: 'AFRICAN CROWNED CRANE',
    1: 'BALD EAGLE',
    2: 'CAMPO FLICKER',
    3: 'GOLDEN EAGLE',
    4: 'KAGU',
}

# Path to the "unknown" species image
unknown_image_path = r'C:\Users\Admin\Downloads\archive\test\AFRICAN PIED HORNBILL\2.jpg'

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img)

    # Use EfficientNet preprocessing if it was used during training
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    st.image(uploaded_file, caption="Uploaded Image", width=200, use_column_width=False)

    if st.button("Predict Bird Species"):
        try:
            predictions = model.predict(img_array).argmax()
            # Check if the prediction is within the known range
            if predictions in map_dict:
                species = map_dict[predictions]
                st.subheader("Prediction:")
                st.write(f"The predicted Bird Species is: {species}")
            else:
                st.subheader("Prediction:")
                st.write("The predicted Bird Species is: Unknown Species")
                # Display the unknown image for unknown species
                st.image(unknown_image_path, caption="Unknown Species", width=200, use_column_width=False)
        except Exception as e:
            st.error(f"Error during prediction: {e}")