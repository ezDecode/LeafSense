# LeafSense - AI-Powered Crop Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LeafSense is a full-stack AI system that detects crop leaf diseases using deep learning, integrates with PlantVillage API for external validation, and enhances results using Google's Gemini API to provide farmer-friendly explanations.

## 🌟 Features

- **AI-Powered Detection**: ResNet50-based deep learning model trained on PlantVillage dataset
- **Multi-API Integration**: Combines local model predictions with PlantVillage API
- **AI Enhancement**: Uses Gemini API to generate farmer-friendly explanations
- **Modern Web Interface**: Responsive, mobile-friendly web UI built with Bootstrap
- **Comprehensive Knowledge Base**: Detailed disease descriptions and treatment recommendations
- **Real-time Processing**: Fast prediction pipeline (≤8 seconds)
- **Scalable Architecture**: Supports multiple concurrent users

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.13+
- Flask 2.3+
- 8GB+ RAM (for model training)
- GPU recommended for training (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ezDecode/LeafSense.git
   cd LeafSense
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your API keys
   GEMINI_API_KEY=your_actual_gemini_api_key
   PLANTVILLAGE_API_KEY=your_actual_plantvillage_api_key
   SECRET_KEY=your_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
LeafSense/
├── app.py                 # Main Flask application
├── train.py              # Model training script
├── train.ipynb           # Jupyter notebook for training
├── requirements.txt      # Python dependencies
├── diseases.json         # Knowledge base of diseases
├── data/                 # Dataset directory
│   ├── train/           # Training images (70%)
│   ├── val/             # Validation images (15%)
│   └── test/            # Test images (15%)
├── saved_models/         # Trained models and artifacts
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── templates/            # HTML templates
└── README.md            # This file
```

## 🧠 Model Training

### Dataset Preparation

1. **Download PlantVillage Dataset**
   - Available on [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
   - Or use the public repository version

2. **Organize Dataset**
   ```
   data/
   ├── train/
   │   ├── Tomato___Late_blight/
   │   ├── Tomato___Early_blight/
   │   └── ... (other classes)
   ├── val/
   └── test/
   ```

3. **Run Training**
   ```bash
   # Using Python script
   python train.py
   
   # Or using Jupyter notebook
   jupyter notebook train.ipynb
   ```

### Model Architecture

- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Head**: Dense(512, ReLU) + Dropout(0.3) + Dense(38, Softmax)
- **Training**: Transfer learning with frozen base layers
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical crossentropy
- **Augmentation**: Rotation, flips, zoom, brightness/contrast

## 🔌 API Endpoints

### POST /predict
Upload an image for disease detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` file (JPG/PNG ≤10MB)

**Response:**
```json
{
  "model_prediction": {
    "disease": "Tomato___Late_blight",
    "confidence": 0.94
  },
  "api_prediction": {
    "disease": "Tomato Late Blight",
    "confidence": 0.92,
    "metadata": "External API prediction"
  },
  "gemini_response": "The leaf shows signs of Tomato Late Blight...",
  "uploaded_image": "filename.jpg",
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🎨 Frontend Features

- **Responsive Design**: Mobile-first approach with Bootstrap 5
- **Drag & Drop**: Easy image upload with preview
- **Real-time Feedback**: Loading states and progress indicators
- **Accessibility**: ARIA labels and keyboard navigation
- **Modern UI**: Clean, farmer-friendly interface

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `PLANTVILLAGE_API_KEY` | PlantVillage API key | Optional |
| `SECRET_KEY` | Flask secret key | Generated |
| `FLASK_ENV` | Flask environment | development |
| `MAX_FILE_SIZE` | Maximum upload size | 10MB |

### Model Configuration

- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 20-30 (with early stopping)
- **Target Accuracy**: ≥90%

## 📊 Performance

- **Prediction Time**: ≤8 seconds
- **Concurrent Users**: 20+
- **Model Accuracy**: ≥90% (target)
- **File Size Limit**: 10MB
- **Supported Formats**: JPG, PNG, JPEG

## 🚀 Deployment

### Production Deployment

1. **Set environment variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your_production_secret
   ```

2. **Use Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **HTTPS Setup** (recommended)
   - Use reverse proxy (Nginx/Apache)
   - SSL certificates (Let's Encrypt)

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive crop disease dataset
- **Google Gemini**: For AI-powered response enhancement
- **TensorFlow/Keras**: For the deep learning framework
- **Flask**: For the web framework
- **Bootstrap**: For the UI components

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ezDecode/LeafSense/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ezDecode/LeafSense/discussions)
- **Wiki**: [Project Wiki](https://github.com/ezDecode/LeafSense/wiki)

## 🔮 Future Enhancements

- [ ] Mobile app with TensorFlow Lite
- [ ] Multi-language support
- [ ] Severity detection (early vs advanced)
- [ ] Geolocation-based outbreak tracking
- [ ] Real-time monitoring and alerts
- [ ] Integration with IoT sensors
- [ ] Advanced analytics dashboard

---

**Made with ❤️ for farmers and agricultural communities worldwide**
