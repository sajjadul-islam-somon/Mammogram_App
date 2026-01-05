import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import plotly.express as px
from PIL import Image
from preprocessing import crop_breast_roi, resize_with_padding

# --- CONFIGURATION ---
EFF_MODEL_PATH = 'best_mammogram_model_phase2_final.keras'
DENSE_MODEL_PATH = 'best_densenet_model.keras'
IMG_SIZE = 224
CLASS_NAMES = ['BI-RADS 1 (Normal)', 'BI-RADS 3 (Benign)', 'BI-RADS 4 (Suspicious)', 'BI-RADS 5 (Malignant)']

# --- 1. PREPROCESSING FUNCTIONS ---
def advanced_preprocessing(image_array):
    """
    Uses the helper functions linked from preprocessing.py
    """
    # 1. Convert to Gray for the cropping algorithm
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array

    # 2. CALL THE LINKED FUNCTION: Smart Cropping
    # This removes artifacts and black borders using logic from preprocessing.py
    img_cropped_gray = crop_breast_roi(img_gray)
    
    # 3. CLAHE (Contrast Enhancement)
    # We apply this here because it works best on Grayscale before resizing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe_gray = clahe.apply(img_cropped_gray)
    
    # 4. Convert back to RGB (Models need 3 channels)
    img_clahe_rgb = cv2.cvtColor(img_clahe_gray, cv2.COLOR_GRAY2RGB)
    
    # 5. CALL THE LINKED FUNCTION: Resize with Padding
    # This resizes the image while keeping the tumor shape correct
    img_final_rgb = resize_with_padding(img_clahe_rgb, (IMG_SIZE, IMG_SIZE))
    
    return img_final_rgb

def preprocess_for_efficientnet(img_array):
    # Get the cleaned, cropped, padded image (0-255)
    img = advanced_preprocessing(img_array)
    # Apply EfficientNet specific scaling
    return tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))

def preprocess_for_densenet(img_array):
    # Get the cleaned, cropped, padded image (0-255)
    img = advanced_preprocessing(img_array)
    # Apply DenseNet specific scaling
    return tf.keras.applications.densenet.preprocess_input(img.astype(np.float32))


# --- 2. XAI FUNCTIONS (Grad-CAM & Score-CAM) ---
def make_gradcam(model, img_array, layer_name):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
    except: return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = tf.gather(preds[0], pred_index)

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def make_scorecam_heatmap(model, img_array, layer_name, top_k=32):
    try:
        # Get activations
        act_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output])
        activations = act_model.predict(img_array, verbose=0)
        
        # Get target class
        preds = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(preds[0])
        
        # Simplified Score-CAM for speed
        # 1. Upsample activations to image size
        upsampled_activations = tf.image.resize(activations, (img_array.shape[1], img_array.shape[2]))
        
        # 2. Normalize
        max_v = tf.math.reduce_max(upsampled_activations, axis=(1,2), keepdims=True)
        min_v = tf.math.reduce_min(upsampled_activations, axis=(1,2), keepdims=True)
        norm_maps = (upsampled_activations - min_v) / (max_v - min_v + 1e-10)
        
        # 3. Compute importance (Forward pass on masked images)
        # We only take the top K most active channels to save time
        channel_sums = tf.reduce_sum(activations, axis=(1,2))[0]
        top_indices = tf.math.top_k(channel_sums, k=min(top_k, activations.shape[-1])).indices
        
        cam = np.zeros((img_array.shape[1], img_array.shape[2]), dtype=np.float32)
        
        for idx in top_indices:
            # Create mask
            mask = norm_maps[0, :, :, idx]
            masked_img = img_array * mask[..., tf.newaxis]
            
            # Predict
            score = model.predict(masked_img, verbose=0)[0][predicted_class]
            cam += score * mask.numpy()
            
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-10)
        return cam
        
    except Exception as e:
        print(f"Score-CAM Error: {e}")
        return None


# --- 3. UI HELPERS ---
def plot_confidence(scores, title):
    fig = px.bar(x=CLASS_NAMES, y=scores, 
                 labels={'x': 'Class', 'y': 'Confidence'},
                 title=title, color=scores, color_continuous_scale='Bluered')
    fig.update_layout(yaxis_range=[0, 1], height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# --- 4. MAIN APP ---
st.set_page_config(page_title="Mammogram Ensemble System", layout="wide")

st.sidebar.title("Diagnostic Settings")
st.sidebar.info("System: Weighted Ensemble (EfficientNet + DenseNet)")
show_explainability = st.sidebar.checkbox("Show Explainability (XAI)", value=True)
xai_method = st.sidebar.radio("Visualization Method:", ["Grad-CAM", "Score-CAM"], disabled=not show_explainability)

st.title("🛡️ AI-Assisted Breast Tumor Classification")
st.markdown("**Dual-Stream Ensemble System:** EfficientNet (Shape) + DenseNet (Texture)")

# Load Models
@st.cache_resource
def load_models():
    def focal_loss_fixed(y_true, y_pred): return 0
    try:
        m1 = tf.keras.models.load_model(EFF_MODEL_PATH, compile=False)
        m2 = tf.keras.models.load_model(DENSE_MODEL_PATH, compile=False, custom_objects={'focal_loss_fixed': focal_loss_fixed})
        return m1, m2
    except Exception as e:
        st.error(f"Error loading models: {e}"); return None, None

eff_net, dense_net = load_models()
uploaded_file = st.file_uploader("Upload Mammogram", type=['png', 'jpg', 'jpeg', 'dcm'])

if uploaded_file and eff_net and dense_net:
    # 1. Process Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    st.image(original_image, caption="Original Scan", width=300)
    
    # 2. Run Predictions
    with st.spinner("Analyzing Shape and Texture features..."):
        # Prepare inputs
        input_eff = np.expand_dims(preprocess_for_efficientnet(original_image), axis=0)
        input_dense = np.expand_dims(preprocess_for_densenet(original_image), axis=0)
        
        # Get individual probabilities
        p1 = eff_net.predict(input_eff, verbose=0)[0]
        p2 = dense_net.predict(input_dense, verbose=0)[0]
        
        # Ensemble Logic (0.7 / 0.3)
        p_final = (0.7 * p1) + (0.3 * p2)

    # 3. DISPLAY RESULTS (3-Step View)
    st.divider()
    st.subheader("Diagnostic Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1: EfficientNet ---
    with col1:
        st.info("Step 1: Shape Analysis")
        st.write("**Model:** EfficientNetB0")
        idx1 = np.argmax(p1)
        st.metric(label="Diagnosis", value=CLASS_NAMES[idx1], delta=f"{p1[idx1]*100:.1f}% Conf.")
        st.plotly_chart(plot_confidence(p1, "Shape Probabilities"), use_container_width=True)

    # --- Column 2: DenseNet ---
    with col2:
        st.warning("Step 2: Texture Analysis")
        st.write("**Model:** DenseNet121")
        idx2 = np.argmax(p2)
        st.metric(label="Diagnosis", value=CLASS_NAMES[idx2], delta=f"{p2[idx2]*100:.1f}% Conf.")
        st.plotly_chart(plot_confidence(p2, "Texture Probabilities"), use_container_width=True)

    # --- Column 3: Ensemble (The Final Say) ---
    with col3:
        st.success("Step 3: Final Consensus")
        st.write("**Method:** Weighted Ensemble")
        idx_final = np.argmax(p_final)
        
        # Highlight High Risk
        if idx_final >= 2: 
            st.error(f"**FINAL: {CLASS_NAMES[idx_final]}**")
        else:
            st.success(f"**FINAL: {CLASS_NAMES[idx_final]}**")
            
        st.metric(label="Aggregated Confidence", value=f"{p_final[idx_final]*100:.2f}%")
        st.plotly_chart(plot_confidence(p_final, "Ensemble Probability"), use_container_width=True)

    # 4. EXPLAINABILITY
    if show_explainability:
        st.markdown("---")
        st.subheader("3. Visual Explanation (XAI)")
        
        cam_col1, cam_col2 = st.columns(2)
        
        # Auto-find layers
        def find_layer(model, name_part):
            for layer in reversed(model.layers):
                if len(layer.output.shape) == 4 and name_part in layer.name: return layer.name
            return None
        
        layer_eff = find_layer(eff_net, 'eff') or find_layer(eff_net, 'block7a')
        layer_dense = find_layer(dense_net, 'conv5') or find_layer(dense_net, 'relu')
        
        with cam_col1:
            st.write("**EfficientNet View (Shape Focus)**")
            if layer_eff:
                if xai_method == "Grad-CAM":
                    hm = make_gradcam(eff_net, input_eff, layer_eff)
                else:
                    with st.spinner("Computing Score-CAM (EfficientNet)..."):
                        hm = make_scorecam_heatmap(eff_net, input_eff, layer_eff)
                        
                if hm is not None:
                    hm_img = cv2.applyColorMap(np.uint8(255*cv2.resize(hm, (original_image.shape[1], original_image.shape[0]))), cv2.COLORMAP_JET)
                    st.image(cv2.addWeighted(original_image, 0.6, hm_img, 0.4, 0), use_container_width=True, caption=f"{xai_method} on {layer_eff}")
            
        with cam_col2:
            st.write("**DenseNet View (Texture Focus)**")
            if layer_dense:
                if xai_method == "Grad-CAM":
                    hm = make_gradcam(dense_net, input_dense, layer_dense)
                else:
                    with st.spinner("Computing Score-CAM (DenseNet)..."):
                        hm = make_scorecam_heatmap(dense_net, input_dense, layer_dense)
                        
                if hm is not None:
                    hm_img = cv2.applyColorMap(np.uint8(255*cv2.resize(hm, (original_image.shape[1], original_image.shape[0]))), cv2.COLORMAP_VIRIDIS)
                    st.image(cv2.addWeighted(original_image, 0.6, hm_img, 0.4, 0), use_container_width=True, caption=f"{xai_method} on {layer_dense}")