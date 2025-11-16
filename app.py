from flask import Flask, render_template,request,redirect,send_from_directory,url_for,jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease.keras")
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image,target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    # print(prediction)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']
        
        if image.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename
        temp_name = f"temp_{uuid.uuid4().hex}"
        filepath = f"static/images/{temp_name}_{image.filename}"
        image.save(filepath)
        
        # Get prediction
        img = extract_features(f'./{filepath}')
        prediction = model.predict(img)
        predicted_class_index = prediction.argmax()
        confidence = float(prediction.max())
        
        # Get disease information from plant_disease.json
        disease_info = plant_disease[predicted_class_index]
        
        # Prepare response
        response = {
            'uploaded_image': f"{temp_name}_{image.filename}",
            'model_prediction': {
                'disease': disease_info['name'],
                'confidence': confidence,
                'cause': disease_info['cause'],
                'cure': disease_info['cure']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    
if __name__ == "__main__":
    app.run(debug=True)