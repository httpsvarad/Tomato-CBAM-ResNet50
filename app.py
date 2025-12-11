import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Conv2D, Multiply, Add,
    Concatenate, Reshape, GlobalMaxPooling2D, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------
# MODEL ARCHITECTURE (same as training)
# ---------------------------

def channel_attention_module(x):
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)

    shared_dense_1 = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))
    shared_dense_2 = Dense(x.shape[-1], activation="sigmoid", kernel_regularizer=l2(0.0001))

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = Add()([avg_out, max_out])
    channel_attention = Reshape((1, 1, x.shape[-1]))(channel_attention)
    return Multiply()([x, channel_attention])


def spatial_attention_module(x):
    avg_pool = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_attention = Conv2D(
        1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        kernel_regularizer=l2(0.0001),
        name="cbam_spatial_conv",
    )(concat)

    return Multiply()([x, spatial_attention])


def cbam_block(input_feature):
    channel_refined = channel_attention_module(input_feature)
    spatial_refined = spatial_attention_module(channel_refined)
    return spatial_refined


def create_model(img_width, img_height, num_classes):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True

    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.55)(x)
    output = Dense(num_classes, activation="softmax", kernel_regularizer=l2(0.0001))(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# ---------------------------
# SETTINGS & WEIGHTS
# ---------------------------

IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 10

# change path if your file name is different
WEIGHTS_PATH = os.path.join("model", "tomato_cbam_model.h5")
st.set_page_config(page_title="Tomato Disease Detection", layout="centered")

st.title("Tomato Disease Detection - with Grad-CAM")
st.write("ResNet50 + CBAM model - Upload an image and optionally view Grad-CAM heatmap.")

# Build and load weights
model = create_model(IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)
try:
    model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
    st.success("✅ Weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading weights: {e}")

# ---------------------------
# Grad-CAM utilities
# ---------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: (1,H,W,3) preprocessed input
    model: Keras model
    last_conv_layer_name: name of the conv layer to target (e.g. 'cbam_spatial_conv')
    pred_index: class index to compute Grad-CAM for; if None uses model's top pred
    returns: heatmap as 2D numpy array (H', W') normalized 0..1
    """
    # 1) Create a model that maps the input image to the activations
    #    of the last conv layer and the model's predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    if last_conv_layer is None:
        raise ValueError(f"Could not find layer {last_conv_layer_name} in the model")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # 2) Compute the gradient of the top predicted class (or pred_index) w.r.t. conv layer outputs
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # compute gradients of class output wrt conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # 3) Pool the gradients over the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4) Multiply each channel in the feature map array by "how important this channel is"
    conv_outputs = conv_outputs[0]  # shape (h, w, channels)
    pooled_grads = pooled_grads.numpy()
    conv_outputs = conv_outputs.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # 5) The channel-wise mean of the resulting feature map is the heatmap
    heatmap = np.mean(conv_outputs, axis=-1)

    # 6) Relu and normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap_on_image(img: Image.Image, heatmap, alpha=0.4, colormap="jet"):
    """
    img : PIL Image (RGB)
    heatmap: 2D numpy array normalized 0..1
    """
    # Resize heatmap to image size
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (img.height, img.width)).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # Convert heatmap to RGBA using matplotlib colormap
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(heatmap_resized)  # (H,W,4) float 0..1

    # Convert to 8bit RGBA
    heatmap_rgba = np.uint8(255 * heatmap_rgba)
    heatmap_img = Image.fromarray(heatmap_rgba).convert("RGBA")

    # Convert original image to RGBA
    img_rgba = img.convert("RGBA")

    # Blend images
    blended = Image.blend(img_rgba, heatmap_img, alpha=alpha)
    return blended, heatmap_img

# ---------------------------
# CLASS NAMES & PREPROCESS
# ---------------------------
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

# ---------------------------
# STREAMLIT UI
# ---------------------------

uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png"])
show_gc = st.checkbox("Show Grad-CAM heatmap", value=True)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict & Explain"):
        with st.spinner("Running model..."):
            x = preprocess_image(img)
            preds = model.predict(x)
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds) * 100)

        st.success(f"Prediction: **{CLASS_NAMES[class_idx]}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        if show_gc:
            try:
                # compute heatmap (uses cbam_spatial_conv as the target layer)
                heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name="cbam_spatial_conv", pred_index=class_idx)
                blended, heatmap_img = overlay_heatmap_on_image(img, heatmap, alpha=0.45, colormap="jet")

                st.write("### Grad-CAM")
                cols = st.columns(3)
                cols[0].image(img, caption="Original", use_container_width=True)
                cols[1].image(heatmap_img, caption="Heatmap (RGBA)", use_container_width=True)
                cols[2].image(blended, caption="Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")
                st.exception(e)