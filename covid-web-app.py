import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.cm as cm
import os
import shutil
import requests

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
    st.write("#### Gradient-weighted Class Activation Mapping *(Grad-CAM)* Visualization")
    st.image(img, use_column_width=True)

# Function to generate model directory structure.
def generate_dirs():
    # Create model's main directory.
    os.mkdir("./covid-models/model")
    os.mkdir("./covid-models/model/variables")
    st.write("## Log")
    st.write("`Generated the directory structure of the model.`")

# Function for removing the model.
def clean_dirs():
    shutil.rmtree("./covid-models/model", ignore_errors=True)
    st.write("`Removed the saved model directory.`")

# Download, extract, and load the model.
def load_model(model_type):
    if model_type == 1:
        return tf.keras.models.load_model('./covid-models/cxr-covid-detection-model-vgg16-m')
    
    # Generate model directory structure.
    if not os.path.isdir("./covid-models/model"):
        generate_dirs()

    saved_model_pb_url = ""
    variables_index_url = ""
    variables_data_url = ""    

    if model_type == 1:
        # VGG-16 (2 Classes).
        saved_model_pb_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-vgg16-m/saved_model.pb"
        variables_index_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-vgg16-m/variables/variables.index"
        variables_data_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-vgg16-m/variables/variables.data-00000-of-00001"
    elif model_type == 2:
        # DenseNet201 (2 Classes).
        saved_model_pb_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-deepnet201/saved_model.pb"
        variables_index_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-deepnet201/variables/variables.index"
        variables_data_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-deepnet201/variables/variables.data-00000-of-00001"
    elif model_type == 3:
        # VGG-16 (4 Classes).
        saved_model_pb_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-vgg16/saved_model.pb"
        variables_index_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-vgg16/variables/variables.index"
        variables_data_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-vgg16/variables/variables.data-00000-of-00001"
    elif model_type == 4:
        # DenseNet201 (4 Classes).
        saved_model_pb_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-densenet201/saved_model.pb"
        variables_index_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-densenet201/variables/variables.index"
        variables_data_url = "https://github.com/srihk/covid-19-detection-from-chest-xray/raw/main/saved-models/cxr-covid-detection-model-4-classes-densenet201/variables/variables.data-00000-of-00001"

    r = requests.get(saved_model_pb_url, allow_redirects=True)
    open('./covid-models/model/saved_model.pb', 'wb').write(r.content)
    st.write("`Loaded ./covid-models/model/saved_model.pb.`")
        
    r = requests.get(variables_index_url, allow_redirects=True)
    open('./covid-models/model/variables/variables.index', 'wb').write(r.content)
    st.write("`Loaded ./covid-models/model/variables/variables.index.`")

    r = requests.get(variables_data_url, allow_redirects=True)
    open('./covid-models/model/variables/variables.data-00000-of-00001', 'wb').write(r.content)
    st.write("`Loaded ./covid-models/model/variables/variables.data-00000-of-00001.`")

    model = tf.keras.models.load_model('./covid-models/model')
    msg = "`Loaded the model of type " + str(model_type) + ".`"
    st.write(msg)

    clean_dirs()
    return model

st.write("# COVID-19 Detection from Chest X-Ray *(CXR)* Scans")

st.write("This project attempts to detect COVID-19 from Chest X-Ray *(CXR)* scans using Transfer Learning (VGG-16).")

# Load the model.
# model = tf.keras.models.load_model('./covid-models/cxr-covid-detection-model-vgg16-m')
# Model Types: 1 - VGG-16 (2 Classes); 2 - DenseNet201 (2 Classes); 3 - VGG-16 (4 Classes); 4 - DenseNet201 (4 Classes)
st.write("## Model Types")
st.write("- 1: VGG-16 (Classes: COVID, Normal)")
st.write("- 2: DenseNet201 (Classes: COVID, Normal)")
st.write("- 3: VGG-16 (Classes: COVID, Normal, Viral Pneumonia, Lung Opacity)")
st.write("- 4: DenseNet201 (Classes: COVID, Normal, Viral Pneumonia, Lung Opacity)")

model_type = 0
model_type = st.number_input("Choose a model type:", min_value=1, max_value=4, value=1, step=1)
model_type = int(model_type)

model_1 = None
model_2 = None
model_3 = None
model_4 = None

if model_type > 4 or model_type < 1:
    model_type = 0
if model_type == 1:
    model_1 = load_model(1)
elif model_type == 2:
    model_2 = load_model(2)
elif model_type == 3:
    model_3 = load_model(3)
elif model_type == 4:
    model_4 = load_model(4)

if model_type <= 4 and model_type >= 1:
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

        # -----------------------------------------------------------------------------------------------------
        # VGG-16 (2 Classes)
        # -----------------------------------------------------------------------------------------------------

        if (model_type == 1):
            st.write("### VGG-16 (2 Classes)")
            prediction = import_and_predict(img, model_1)

            msg = "COVID-19 **+VE** *(POSITIVE)*: " + str(prediction[0][0] * 100) + "%"
            st.write(msg)

            msg = "COVID-19 **-VE** *(NEGATIVE)* " + str(prediction[0][1] * 100) + "%"
            st.write(msg)

            st.write("Raw results (0: **COVID**, 1: **Normal**)")
            st.write(prediction)

            # Grad-CAM Visualization.
            img_size = (224, 224)
            last_conv_layer_name = "block5_conv3"
            img_path = 'img.png'

            img_arr = preprocess_input(get_img_array(img_path, size=img_size))

            # Remove activation for the outermost layer.
            model_1.layers[-1].activation = None

            # Generate Grad-CAM heatmap.
            heatmap = make_gradcam_heatmap(img_arr, model_1, last_conv_layer_name)

            # Display heatmap.
            save_and_display_gradcam(img_path, heatmap)

        # -----------------------------------------------------------------------------------------------------
        # DenseNet201 (2 Classes)
        # -----------------------------------------------------------------------------------------------------

        if model_type == 2:
            st.write("### DenseNet201 (2 Classes)")
            prediction = import_and_predict(img, model_2)

            msg = "COVID-19 **+VE** *(POSITIVE)*: " + str(prediction[0][0] * 100) + "%"
            st.write(msg)

            msg = "COVID-19 **-VE** *(NEGATIVE)* " + str(prediction[0][1] * 100) + "%"
            st.write(msg)

            st.write("Raw results (0: **COVID**, 1: **Normal**)")
            st.write(prediction)

            # Grad-CAM Visualization.
            img_size = (224, 224)
            last_conv_layer_name = "conv5_block32_concat"
            img_path = 'img.png'

            img_arr = preprocess_input(get_img_array(img_path, size=img_size))

            # Remove activation for the outermost layer.
            model_2.layers[-1].activation = None

            # Generate Grad-CAM heatmap.
            heatmap = make_gradcam_heatmap(img_arr, model_2, last_conv_layer_name)

            # Display heatmap.
            save_and_display_gradcam(img_path, heatmap)

        # -----------------------------------------------------------------------------------------------------
        # VGG-16 (4 Classes)
        # -----------------------------------------------------------------------------------------------------

        if model_type == 3:
            st.write("### VGG-16 (4 Classes)")
            prediction = import_and_predict(img, model_3)

            msg = "COVID-19: " + str(prediction[0][0] * 100) + "%"
            st.write(msg)

            msg = "Normal: " + str(prediction[0][1] * 100) + "%"
            st.write(msg)

            msg = "Viral Pneumonia: " + str(prediction[0][2] * 100) + "%"
            st.write(msg)

            msg = "Lung Opacity: " + str(prediction[0][3] * 100) + "%"
            st.write(msg)

            st.write("Raw results (0: **COVID**, 1: **Normal**, 2: **Viral Pneumonia**, 3: **Lung Opacity**)")
            st.write(prediction)

            # Grad-CAM Visualization.
            img_size = (224, 224)
            last_conv_layer_name = "block5_conv3"
            img_path = 'img.png'

            img_arr = preprocess_input(get_img_array(img_path, size=img_size))

            # Remove activation for the outermost layer.
            model_3.layers[-1].activation = None

            # Generate Grad-CAM heatmap.
            heatmap = make_gradcam_heatmap(img_arr, model_3, last_conv_layer_name)

            # Display heatmap.
            save_and_display_gradcam(img_path, heatmap)

        # -----------------------------------------------------------------------------------------------------
        # DenseNet201 (4 Classes)
        # -----------------------------------------------------------------------------------------------------

        if model_type == 4:
            st.write("### DenseNet201 (4 Classes)")
            prediction = import_and_predict(img, model_4)

            msg = "COVID-19: " + str(prediction[0][0] * 100) + "%"
            st.write(msg)

            msg = "Normal: " + str(prediction[0][1] * 100) + "%"
            st.write(msg)

            msg = "Viral Pneumonia: " + str(prediction[0][2] * 100) + "%"
            st.write(msg)

            msg = "Lung Opacity: " + str(prediction[0][3] * 100) + "%"
            st.write(msg)

            st.write("Raw results (0: **COVID**, 1: **Normal**, 2: **Viral Pneumonia**, 3: **Lung Opacity**)")
            st.write(prediction)

            # Grad-CAM Visualization.
            img_size = (224, 224)
            last_conv_layer_name = "conv5_block32_concat"
            img_path = 'img.png'

            img_arr = preprocess_input(get_img_array(img_path, size=img_size))

            # Remove activation for the outermost layer.
            model_4.layers[-1].activation = None

            # Generate Grad-CAM heatmap.
            heatmap = make_gradcam_heatmap(img_arr, model_4, last_conv_layer_name)

            # Display heatmap.
            save_and_display_gradcam(img_path, heatmap)