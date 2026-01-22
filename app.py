"""
Wine Cultivar Origin Prediction System
Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# ROBUST MODEL PATH - Anchored to file location (CRITICAL FIX)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl')

try:
    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    scaler = model_package['scaler']
    features = model_package['features']
    accuracy = model_package['accuracy']
    algorithm = model_package['algorithm']
    print(f"✓ Model loaded successfully!")
    print(f"  Algorithm: {algorithm}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Features: {features}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Cultivar names for better user experience
CULTIVAR_NAMES = {
    0: "Cultivar 0",
    1: "Cultivar 1", 
    2: "Cultivar 2"
}

# Feature information for the form with validation ranges
FEATURE_INFO = {
    'alcohol': {
        'label': 'Alcohol',
        'unit': '%',
        'min': 11.0,
        'max': 15.0,
        'step': 0.1,
        'default': 13.0
    },
    'malic_acid': {
        'label': 'Malic Acid',
        'unit': 'g/L',
        'min': 0.5,
        'max': 6.0,
        'step': 0.1,
        'default': 2.0
    },
    'total_phenols': {
        'label': 'Total Phenols',
        'unit': 'mg/L',
        'min': 0.5,
        'max': 4.0,
        'step': 0.1,
        'default': 2.0
    },
    'flavanoids': {
        'label': 'Flavanoids',
        'unit': 'mg/L',
        'min': 0.0,
        'max': 6.0,
        'step': 0.1,
        'default': 2.0
    },
    'color_intensity': {
        'label': 'Color Intensity',
        'unit': '',
        'min': 1.0,
        'max': 13.0,
        'step': 0.1,
        'default': 5.0
    },
    'proline': {
        'label': 'Proline',
        'unit': 'mg/L',
        'min': 200,
        'max': 1700,
        'step': 10,
        'default': 700
    }
}


@app.route('/')
def home():
    """Render the home page with input form"""
    if model is None:
        return "Error: Model not loaded. Please ensure wine_cultivar_model.pkl exists in the model folder.", 500
    
    return render_template('index.html', 
                         features=features,
                         feature_info=FEATURE_INFO,
                         algorithm=algorithm,
                         accuracy=accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with proper error handling"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get input data from form with PROPER MISSING FIELD HANDLING (CRITICAL FIX)
        input_data = []
        input_display = {}
        
        for feature in features:
            # Use request.form.get() with explicit missing-field error handling
            value_raw = request.form.get(feature)
            
            # Check if field is missing
            if value_raw is None:
                return jsonify({
                    'success': False,
                    'error': f'{feature} is required'
                }), 400
            
            # Validate and convert to float
            try:
                value = float(value_raw)
                
                # Validate range
                if feature in FEATURE_INFO:
                    min_val = FEATURE_INFO[feature]['min']
                    max_val = FEATURE_INFO[feature]['max']
                    if value < min_val or value > max_val:
                        return jsonify({
                            'success': False,
                            'error': f'{feature} must be between {min_val} and {max_val}'
                        }), 400
                
                input_data.append(value)
                input_display[feature] = value
                
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f'{feature} must be a valid number'
                }), 400
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input data
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = float(prediction_proba[prediction]) * 100
        
        # Get cultivar name
        cultivar_name = CULTIVAR_NAMES.get(prediction, f"Cultivar {prediction}")
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'cultivar_name': cultivar_name,
            'confidence': round(confidence, 2),
            'input_data': input_display,
            'probabilities': {
                CULTIVAR_NAMES[i]: round(float(prob) * 100, 2) 
                for i, prob in enumerate(prediction_proba)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/health')
def health():
    """Health check endpoint for monitoring deployments"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'algorithm': algorithm if model else None,
        'accuracy': float(accuracy) if model else None
    })


if __name__ == '__main__':
    # Get port from environment variable (for deployment) or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # PRODUCTION-SAFE DEBUG TOGGLE (CRITICAL FIX)
    # Use environment variable to control debug mode
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    # Run the app with gunicorn in production
    app.run(host='0.0.0.0', port=port, debug=debug)