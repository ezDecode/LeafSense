# LeafSense - AI-Powered Crop Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)](https://tensorflow.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![cuDNN](https://img.shields.io/badge/cuDNN-9.0-blue.svg)](https://developer.nvidia.com/cudnn)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LeafSense is a full-stack AI system that detects crop leaf diseases using deep learning, integrates with PlantVillage API for external validation, and enhances results using Google's Gemini API to provide farmer-friendly explanations.

## 🌟 Features

- **AI-Powered Detection**: EfficientNetB4-based deep learning model with transfer learning
- **Multi-API Integration**: Combines local model predictions with PlantVillage API
- **AI Enhancement**: Uses Gemini API to generate farmer-friendly explanations
- **Modern Web Interface**: Responsive, mobile-friendly web UI built with Bootstrap
- **Comprehensive Knowledge Base**: Detailed disease descriptions and treatment recommendations
- **Real-time Processing**: Fast prediction pipeline (≤8 seconds)
- **Scalable Architecture**: Supports multiple concurrent users
- **GPU Support**: Optimized for CUDA 13 and cuDNN 9
- **Simple Training**: Ultra-simplified 3-step training process

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (recommended)
- **TensorFlow 2.16.1+** (with CUDA support for GPU)
- **Flask 3.0.3+**
- **8GB+ RAM** (for model training)
- **NVIDIA GPU** (optional, for faster training)
- **Kaggle Account** (free - for dataset download)

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

4. **Download the pre-trained model**
   
   The trained model file (`plant_disease.keras`) is not included in the repository due to its size (203 MB).
   
   **Option A: Download from Google Drive**
   - Download the model from: [Google Drive Link](https://drive.google.com/your-link-here)
   - Place it in the `models/` directory
   - File path should be: `models/plant_disease.keras`
   
   **Option B: Train your own model**
   - Follow the training instructions in the [Model Training](#-model-training) section below
   - The training script will save the model to `models/plant_disease.keras`

5. **Set up Kaggle API (for dataset download)**
   
   a. Create a free account at [kaggle.com](https://www.kaggle.com/)
   
   b. Get your API token:
   - Go to `https://www.kaggle.com/settings`
   - Scroll to the **API** section
   - Click **"Create New API Token"**
   - This downloads `kaggle.json`
   
   c. Place the token file:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   
   d. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

6. **Set up environment variables (optional - for API integrations)**
   ```bash
   # Copy the example environment file (if exists)
   cp .env.example .env
   
   # Edit .env with your API keys (optional for basic usage)
   GEMINI_API_KEY=your_actual_gemini_api_key  # Optional
   PLANTVILLAGE_API_KEY=your_actual_plantvillage_api_key  # Optional
   SECRET_KEY=your_secret_key_here
   ```

7. **Prepare your dataset (if training)**
   - Organize images in folders by disease class
   - Expected structure: `data/Disease_Name/images.jpg`
   - Or download PlantVillage dataset manually from Kaggle

8. **Train the model (optional - skip if using pre-trained model)**
   ```bash
   python train.py
   # Default: loads from 'data/', trains 10 epochs, saves to 'models/'
   ```

9. **Run the application**
   ```bash
   python app.py
   ```

10. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
LeafSense/
├── app.py                 # Main Flask application
├── train.py              # Simplified model training script (80 lines)
├── requirements.txt      # Python dependencies
├── plant_disease.json    # Knowledge base of diseases
├── data/                 # Dataset directory (organized by class)
│   ├── Disease_Class_1/
│   ├── Disease_Class_2/
│   └── ... (auto-split 80/20 train/val)
├── models/               # Trained models (excluded from Git - too large)
│   └── plant_disease.keras # Main model (download separately)
├── static/               # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── templates/            # HTML templates
└── README.md            # This file
```

## 🧠 Model Training

### Dataset Setup

**Option 1: Use Your Own Data**

Organize images in folders by disease class:
```
data/
├── Tomato_Late_Blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Tomato_Early_Blight/
│   └── ...
└── Potato_Healthy/
    └── ...
```

**Option 2: Download PlantVillage Dataset**

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size**: ~2 GB
- **Images**: 87,000+ augmented images
- **Classes**: 38 plant disease categories

### Training Process (3 Simple Steps)

1. **Run the training script**
   ```bash
   python train.py
   ```

2. **What happens:**
   - ✅ Loads images from `data/` folder
   - ✅ Automatically splits 80/20 train/validation
   - ✅ Builds EfficientNetB4 model with transfer learning
   - ✅ Trains for 10 epochs (customizable)
   - ✅ Saves model to `saved_models/best_model.keras`
   - ✅ Saves class mapping to `saved_models/class_indices.json`

3. **Customize (optional)**
   Edit the bottom of `train.py`:
   ```python
   train_and_save(
       data_dir='data',           # Your data folder
       epochs=20,                 # More training
       output_dir='saved_models'  # Output location
   )
   ```

### Model Architecture

- **Base Model**: EfficientNetB4 pre-trained on ImageNet (frozen)
- **Custom Head**: GlobalAvgPool → Dropout(0.3) → Dense(256, ReLU) → Dense(num_classes, Softmax)
- **Training**: Transfer learning - only custom head is trained
- **Optimizer**: Adam
- **Loss**: Sparse categorical crossentropy
- **Image Size**: 224×224
- **Batch Size**: 32
- **Normalization**: Pixels scaled to 0-1 range

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

### Training Configuration

- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (adjustable)
- **Epochs**: 10 (default, customizable)
- **Validation Split**: 20% automatic
- **Expected Accuracy**: 85-90%
- **Training Time**: 2-3 min/epoch (GPU), 20-30 min/epoch (CPU)

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
