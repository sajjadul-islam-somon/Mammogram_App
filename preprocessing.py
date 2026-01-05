
import cv2
import numpy as np

# Configuration
IMG_SIZE = 224

def crop_breast_roi(img_gray):
    """
    Step 1: Artifact Removal & ROI Cropping.
    - Thresholds to find tissue.
    - Finds largest contour (breast).
    - Crops to bounding box.
    """
    # Threshold (simple binary to separate tissue from background)
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_gray  # Safety: Return original if fails

    # Find the largest contour (The Breast)
    c = max(contours, key=cv2.contourArea)
    
    # Get Bounding Box
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop
    cropped_img = img_gray[y:y+h, x:x+w]
    return cropped_img

def resize_with_padding(img, target_size=(224, 224)):
    """
    Step 2: Resize while maintaining Aspect Ratio (No Stretching).
    - Scales image to fit within target_size.
    - Adds black padding to fill the gaps.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale to fit the largest dimension
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize to the fitted size
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate centering position
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste the resized image onto the center of the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def preprocess_image(img_array, target_size=(224, 224)):
    """
    Master Pipeline for Deployment (Now with Padding).
    Input: Expects a numpy array (RGB)
    Output: Normalized (0-1) numpy array
    """
    # 1. Convert to Gray for cropping logic
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # 2. Smart Cropping (Remove original black borders)
    img_cropped_gray = crop_breast_roi(img_gray)
    
    # 3. CLAHE (Contrast Enhancement on Gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe_gray = clahe.apply(img_cropped_gray)
    
    # 4. Convert back to RGB (Models need 3 channels)
    img_clahe_rgb = cv2.cvtColor(img_clahe_gray, cv2.COLOR_GRAY2RGB)
    
    # 5. Resize with Padding (PRESERVES ASPECT RATIO)
    img_final_rgb = resize_with_padding(img_clahe_rgb, target_size)
    
    # 6. Normalize (0-1)
    img_normalized = img_final_rgb / 255.0
    
    return img_normalized
