# 🍷 Wine Cultivar Origin Prediction System

A machine learning web application that predicts wine cultivar (origin/class) based on chemical properties using the Wine Dataset.

## 📋 Project Overview

This project implements a complete end-to-end machine learning pipeline:
- **Data preprocessing** with feature selection and scaling
- **Random Forest Classifier** for multiclass classification
- **Flask web application** with modern UI
- **Production deployment** on Render.com

## 🎯 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~97%+ on test data
- **Features Used**: 6 chemical properties
  - Alcohol (%)
  - Malic Acid (g/L)
  - Total Phenols (mg/L)
  - Flavanoids (mg/L)
  - Color Intensity
  - Proline (mg/L)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/WineCultivar_Project_YourName_MatricNo.git
   cd WineCultivar_Project_YourName_MatricNo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
WineCultivar_Project_YourName_MatricNo/
├── app.py                                  # Flask application (inference)
├── requirements.txt                        # Python dependencies
├── WineCultivar_hosted_webGUI_link.txt    # Submission information
├── README.md                               # Project documentation
├── .gitignore                              # Git ignore rules
├── model/
│   ├── model_building.ipynb                # Training & evaluation code
│   └── wine_cultivar_model.pkl             # Trained model (Joblib)
├── static/
│   └── style.css                           # Frontend styling
└── templates/
    └── index.html                          # Web interface
```

## 🔬 Model Development Process

The model development is documented in `model/model_building.ipynb`:

1. **Data Loading**: Load Wine dataset from sklearn
2. **Preprocessing**: 
   - Check for missing values
   - Select 6 features from available 13
   - Split data (80% train, 20% test)
   - Apply StandardScaler for feature normalization
3. **Model Training**: 
   - Random Forest Classifier with 100 estimators
   - Hyperparameters: max_depth=10, min_samples_split=5
4. **Evaluation**:
   - Accuracy Score
   - Precision, Recall, F1-Score (Macro & Weighted)
   - Classification Report
   - Confusion Matrix
   - Feature Importance Analysis
5. **Model Persistence**: Save using Joblib

## 🌐 API Endpoints

### `GET /`
Renders the main prediction interface

### `POST /predict`
Makes predictions based on input features

**Request Body** (form data):
```
alcohol: float (11.0-15.0)
malic_acid: float (0.5-6.0)
total_phenols: float (0.5-4.0)
flavanoids: float (0.0-6.0)
color_intensity: float (1.0-13.0)
proline: float (200-1700)
```

**Response** (JSON):
```json
{
  "success": true,
  "prediction": 0,
  "cultivar_name": "Cultivar 0",
  "confidence": 98.50,
  "probabilities": {
    "Cultivar 0": 98.50,
    "Cultivar 1": 1.25,
    "Cultivar 2": 0.25
  },
  "input_data": {...}
}
```

### `GET /health`
Health check endpoint for monitoring

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "algorithm": "Random Forest Classifier",
  "accuracy": 0.9722
}
```

## 🎨 Features

- ✅ Modern, responsive UI with gradient design
- ✅ Real-time form validation
- ✅ Confidence scores with visual progress bars
- ✅ Class probability visualization
- ✅ Input summary display
- ✅ Error handling and user feedback
- ✅ Mobile-friendly interface

## 🔧 Technologies Used

- **Backend**: Flask 3.0.0
- **ML Framework**: scikit-learn 1.3.2
- **Data Processing**: pandas, numpy
- **Model Persistence**: Joblib
- **Deployment**: Gunicorn (WSGI server)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## 📊 Model Evaluation Metrics

The model achieves excellent performance:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~97%+
- **Weighted F1-Score**: ~0.97
- **Macro F1-Score**: ~0.97

Detailed metrics available in the Jupyter notebook.

## 🚀 Deployment

### Deploy to Render.com

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: Python 3
5. Deploy!

### Environment Variables (Optional)

- `PORT`: Port number (default: 5000)
- `FLASK_DEBUG`: Set to '1' for debug mode (not recommended in production)

## 📝 License

This project is for educational purposes as part of a Machine Learning course assignment.

## 👤 Author

- **Name**: [Your Name]
- **Matric Number**: [Your Matric Number]
- **Course**: Machine Learning / AI

## 🙏 Acknowledgments

- Wine Dataset from UCI Machine Learning Repository
- scikit-learn library for ML tools
- Flask framework for web development

---

**Note**: This project demonstrates understanding of:
- Data preprocessing pipelines
- Machine learning algorithm implementation
- Model evaluation and validation
- Web application development
- Production deployment practices