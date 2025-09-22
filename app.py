import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load knowledge base
def load_knowledge_base():
    try:
        with open('diseases.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("diseases.json not found, using empty knowledge base")
        return {}

# Load trained model
def load_model():
    try:
        # Try new format first, fallback to old format
        model_path = 'saved_models/best_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully (.keras format)")
            return model
        
        # Fallback to old format
        old_model_path = 'saved_models/best_model.h5'
        if os.path.exists(old_model_path):
            model = tf.keras.models.load_model(old_model_path)
            logger.info("Model loaded successfully (.h5 format)")
            return model
            
        logger.warning("Model file not found, returning None")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load class mapping
def load_class_mapping():
    try:
        with open('saved_models/class_indices.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("class_indices.json not found")
        return {}

# Initialize global variables
model = load_model()
class_mapping = load_class_mapping()
knowledge_base = load_knowledge_base()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def predict_local_model(image_array):
    """Predict using local trained model"""
    if model is None:
        return None
    
    try:
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get class name from mapping
        class_name = class_mapping.get(str(predicted_class), f"Class_{predicted_class}")
        
        return {
            "disease": class_name,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error in local model prediction: {e}")
        return None

def call_plantvillage_api(image_path):
    """Call PlantVillage API for external prediction"""
    try:
        # This is a placeholder - you'll need to replace with actual PlantVillage API endpoint
        # For now, we'll simulate an API call
        api_url = os.getenv('PLANTVILLAGE_API_URL', 'https://api.plantvillage.psu.edu/v1')
        
        # Simulate API response (replace with actual API call)
        # response = requests.post(api_url, files={'image': open(image_path, 'rb')})
        
        # Placeholder response
        return {
            "disease": "Tomato Late Blight",
            "confidence": 0.92,
            "metadata": "External API prediction"
        }
    except Exception as e:
        logger.error(f"Error calling PlantVillage API: {e}")
        return None

def enhance_with_gemini(local_pred, api_pred, remedies):
    """Enhance response using Gemini API"""
    try:
        if not os.getenv('GEMINI_API_KEY'):
            logger.warning("Gemini API key not found, returning basic response")
            return f"Based on our analysis, the leaf shows signs of {local_pred['disease'] if local_pred else 'unknown disease'}. {remedies}"
        
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        You are an agricultural assistant helping farmers understand crop diseases.
        
        Based on these predictions:
        - Local Model: {local_pred}
        - External API: {api_pred}
        - Available Remedies: {remedies}
        
        Please provide a clear, farmer-friendly response that includes:
        1. Disease name in simple terms
        2. Brief explanation of what this disease is
        3. Practical remedies and prevention measures
        4. Any immediate actions the farmer should take
        
        Keep the language simple and actionable for farmers.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return f"Analysis complete. Disease: {local_pred['disease'] if local_pred else 'Unknown'}. {remedies}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_array = preprocess_image(filepath)
        if image_array is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Get local model prediction
        local_prediction = predict_local_model(image_array)
        
        # Get PlantVillage API prediction
        api_prediction = call_plantvillage_api(filepath)
        
        # Get remedies from knowledge base
        disease_name = local_prediction['disease'] if local_prediction else 'Unknown'
        remedies = knowledge_base.get(disease_name, {}).get('remedy', 'General care recommended')
        
        # Enhance with Gemini API
        gemini_response = enhance_with_gemini(local_prediction, api_prediction, remedies)
        
        # Prepare response
        result = {
            'model_prediction': local_prediction,
            'api_prediction': api_prediction,
            'gemini_response': gemini_response,
            'uploaded_image': filename,
            'timestamp': str(np.datetime64('now'))
        }
        
        logger.info(f"Prediction completed successfully for {filename}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
