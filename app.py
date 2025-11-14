"""LeafSense - Plant Disease Detection"""
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def load_knowledge_base():
    """Load disease information from JSON"""
    try:
        with open('diseases.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def load_model():
    """Load trained model"""
    try:
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Try loading model
        for path in ['saved_models/best_model.keras', 'saved_models/best_model.h5']:
            if Path(path).exists():
                return tf.keras.models.load_model(path)
        return None
    except:
        return None

def load_class_mapping():
    """Load class index to name mapping"""
    try:
        with open('saved_models/class_indices.json', 'r') as f:
            return json.load(f)
    except:
        return {}

# Initialize global variables
model = load_model()
class_mapping = load_class_mapping()
knowledge_base = load_knowledge_base()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except:
        return None

def predict(image_array):
    """Make prediction using loaded model"""
    if model is None:
        return None
    try:
        predictions = model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        disease = class_mapping.get(str(predicted_class), f"Unknown_{predicted_class}")
        return {"disease": disease, "confidence": confidence}
    except:
        return None



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if not file.filename or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save image
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        # Preprocess and predict
        image_array = preprocess_image(filepath)
        if image_array is None:
            return jsonify({'error': 'Image processing failed'}), 500
        
        prediction = predict(image_array)
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get disease info
        disease_name = prediction['disease']
        disease_info = knowledge_base.get(disease_name, {})
        
        return jsonify({
            'disease': disease_name,
            'confidence': prediction['confidence'],
            'description': disease_info.get('description', 'No information available'),
            'remedy': disease_info.get('remedy', 'Consult agricultural expert'),
            'prevention': disease_info.get('prevention', 'Follow good practices'),
            'uploaded_image': filename
        })
    except:
        return jsonify({'error': 'Server error'}), 500

@app.route('/static/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)