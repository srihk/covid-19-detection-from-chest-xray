import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.cm as cm

# Function to fetch an image in the form of an array.
def import_and_predict(img, model):
    arr_img = image.img_to_array(img)
    arr_img = np.expand_dims(arr_img, axis = 0)

    # Preprocess the input using VGG-16's preprocess_input() method.
    arr_img_processed = preprocess_input(arr_img)
    
    # Perform prediction.
    result = model.predict(arr_img_processed)
    return result

# Function to fetch an image in the form of an array.
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size = size)
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# Function to generate the Grad-CAM heatmap from https://keras.io/examples/vision/grad_cam/.
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer.
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer.
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation.
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1.
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    img = image.load_img(cam_path, target_size = size)
    st.write("## Gradient-weighted Class Activation Mapping *(Grad-CAM)* Visualization")
    st.image(img, use_column_width=True)

# Load the model.
model = tf.keras.models.load_model('./covid-models/cxr-covid-detection-model-max-pooling-10-epochs')

st.write("# COVID-19 Detection from Chest X-Ray *(CXR)* Scans")

st.write("This project attempts to detect COVID-19 from Chest X-Ray *(CXR)* scans using Transfer Learning (VGG-16).")

file = st.file_uploader("Please upload a Chest X-Ray (CXR) Scan image file.", type=["png"])

if file is not None:
    size = (224, 224)
    byte_file = file.read()
    with open('img.png', mode='wb+') as f:
        f.write(byte_file)
    img = image.load_img("img.png", target_size = size)
    st.write("## Original Image")
    st.image(img, use_column_width=True)
    
    st.write("## Prediction Results")
    prediction = import_and_predict(img, model)

    if np.argmax(prediction) == 0:
        st.write("Status: COVID-19 **+VE** *(POSITIVE)*")
    elif np.argmax(prediction) == 1:
        st.write("Status: COVID-19 **-VE** *(NEGATIVE)*")

    st.write("Probability (0: **+VE**, 1: **-VE**)")
    st.write(prediction)

    # Grad-CAM Visualization.
    img_size = (224, 224)
    last_conv_layer_name = "block5_conv3"
    img_path = 'img.png'

    img_arr = preprocess_input(get_img_array(img_path, size=img_size))

    # Remove activation for the outermost layer.
    model.layers[-1].activation = None

    # Generate Grad-CAM heatmap.
    heatmap = make_gradcam_heatmap(img_arr, model, last_conv_layer_name)

    # Display heatmap.
    save_and_display_gradcam(img_path, heatmap)