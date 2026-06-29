import numpy as np
from PIL import Image
import io
import os
import random

# Try to import TensorFlow, fallback to Mock if missing
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Running in MOCK mode.")

# Load model and labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../public/model_unquant.tflite')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '../public/labels.txt')

interpreter = None
labels = []

def load_model():
    global interpreter, labels
    
    # Load labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        print(f"Warning: Labels file not found at {LABELS_PATH}")
        # Default labels if missing
        if not labels:
            labels = ["0 Plastic Bottle", "1 Aluminum Can", "2 Glass Bottle", "3 Paper"]

    if not TF_AVAILABLE:
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Load TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_image(image_bytes):
    global interpreter
    
    if not TF_AVAILABLE or interpreter is None:
        # MOCK INFERENCE
        img = Image.open(io.BytesIO(image_bytes))
        # Simulate processing time
        
        mock_label = random.choice(labels)
        confidence = 0.85 + (random.random() * 0.14)
        
        clean_label = mock_label.split(' ', 1)[1] if ' ' in mock_label else mock_label
        
        return {
            "label": clean_label,
            "confidence": confidence,
            "raw_label": mock_label,
            "note": "MOCK_RESULT_PYTHON_3_14_INCOMPATIBLE"
        }

    # Real Inference
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    input_data = np.array(img, dtype=np.float32)
    
    # Normalize if needed (depends on how model was trained)
    # Usually Teachable Machine models expect 0-1 or -1 to 1 based on metadata
    # Assuming standard 0-255 image, let's normalize to 0-1 first
    input_data = (input_data / 127.5) - 1.0
    input_data = np.expand_dims(input_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # Find top prediction
    top_index = np.argmax(results)
    confidence = float(results[top_index])
    
    label = labels[top_index] if top_index < len(labels) else "Unknown"
    
    # Cleanup label (Teachable Machine often adds "0 ClassName")
    clean_label = label.split(' ', 1)[1] if ' ' in label else label

    return {
        "label": clean_label,
        "confidence": confidence,
        "raw_label": label
    }
