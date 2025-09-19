import os
import torch
import torch.nn as nn
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import base64
from io import BytesIO
from models import CNNModel, DNNModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Class names will be loaded dynamically from the dataset
CLASS_NAMES = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.478, 0.472, 0.430), (0.242, 0.238, 0.258))
])

# Model definitions are now imported from models.py

# Global variables for models
cnn_model = None
dnn_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def load_class_names():
    """Load class names from the dataset"""
    global CLASS_NAMES
    try:
        # Try to load from the CNN training data directory
        from torchvision import datasets
        data_dir = '/Users/cajetan/Documents/Deep_Lerning/Prosit_1/archive'
        if os.path.exists(f'{data_dir}/train'):
            temp_dataset = datasets.ImageFolder(f'{data_dir}/train')
            CLASS_NAMES = temp_dataset.classes
            print(f"Loaded class names: {CLASS_NAMES}")
        else:
            # Fallback to CINIC-10 standard classes
            CLASS_NAMES = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            print("Using fallback CINIC-10 class names")
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to CINIC-10 standard classes
        CLASS_NAMES = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

def load_models():
    """Load both CNN and DNN models"""
    global cnn_model, dnn_model
    
    try:
        # Load CNN model
        cnn_model = CNNModel()
        cnn_path = 'models/cnn_model.pth'
        if os.path.exists(cnn_path):
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
            cnn_model.to(device)
            cnn_model.eval()
            print("CNN model loaded successfully")
        else:
            print(f"CNN model not found at {cnn_path}")
            print("Train the CNN model first by running: python ../CNN/src/cnn_model.py")
            cnn_model = None
            
        # Load DNN model
        dnn_model = DNNModel()
        dnn_path = 'models/dnn_model.pth'
        if os.path.exists(dnn_path):
            dnn_model.load_state_dict(torch.load(dnn_path, map_location=device))
            dnn_model.to(device)
            dnn_model.eval()
            print("DNN model loaded successfully")
        else:
            print(f"DNN model not found at {dnn_path}")
            print("Train the DNN model first by running: python ../DNN/src/dnn_model.py")
            dnn_model = None
            
    except Exception as e:
        print(f"Error loading models: {e}")
        cnn_model = None
        dnn_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_tensor, model_type):
    """Make prediction using specified model with confidence-based flagging"""
    try:
        with torch.no_grad():
            # Ensure image tensor is on the same device as the model
            image_tensor = image_tensor.to(device)
            
            if model_type == 'cnn' and cnn_model is not None:
                outputs = cnn_model(image_tensor)
            elif model_type == 'dnn' and dnn_model is not None:
                outputs = dnn_model(image_tensor)
            else:
                return None, None, None
            
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            results = []
            for i in range(3):
                results.append({
                    'class': CLASS_NAMES[top_indices[i].item()],
                    'confidence': top_probs[i].item() * 100
                })
            
            # Confidence-based reliability assessment
            top_confidence = top_probs[0].item() * 100
            confidence_gap = (top_probs[0].item() - top_probs[1].item()) * 100
            
            # Determine reliability flag based on confidence thresholds
            reliability_flag = get_reliability_flag(top_confidence, confidence_gap, model_type)
            
            return results, CLASS_NAMES[predicted.item()], reliability_flag
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

def get_reliability_flag(top_confidence, confidence_gap, model_type):
    """Determine reliability flag based on confidence metrics"""
    
    # Model-specific thresholds based on actual performance
    if model_type == 'cnn':
        # CNN thresholds (better performing model)
        high_confidence_threshold = 80.0
        moderate_confidence_threshold = 70.0
        min_gap_threshold = 50.0
    else:  # DNN
        # DNN thresholds (lower performing model, more lenient)
        high_confidence_threshold = 70.0
        moderate_confidence_threshold = 60.0
        min_gap_threshold = 20.0
    
    # Determine flag based on confidence and gap
    if top_confidence >= high_confidence_threshold and confidence_gap >= min_gap_threshold:
        return {
            'level': 'high',
            'message': 'High confidence prediction - likely accurate',
            'color': 'success',
            'icon': 'check-circle'
        }
    elif top_confidence >= moderate_confidence_threshold and confidence_gap >= min_gap_threshold * 0.7:
        return {
            'level': 'moderate',
            'message': 'Moderate confidence - prediction may be correct',
            'color': 'warning',
            'icon': 'exclamation-triangle'
        }
    elif top_confidence < moderate_confidence_threshold or confidence_gap < min_gap_threshold * 0.5:
        return {
            'level': 'low',
            'message': 'Low confidence - prediction may be unreliable',
            'color': 'danger',
            'icon': 'x-circle'
        }
    else:
        return {
            'level': 'uncertain',
            'message': 'Uncertain prediction - consider alternative classes',
            'color': 'secondary',
            'icon': 'question-circle'
        }

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_tensor = preprocess_image(filepath)
        if image_tensor is None:
            flash('Error processing image')
            return redirect(url_for('index'))
        
        # Make prediction
        results, predicted_class, reliability_flag = predict_image(image_tensor, model_type)
        if results is None:
            flash(f'{model_type.upper()} model not available')
            return redirect(url_for('index'))
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('results.html', 
                             results=results,
                             predicted_class=predicted_class,
                             model_type=model_type.upper(),
                             image_data=img_base64,
                             reliability_flag=reliability_flag)
    
    flash('Invalid file type. Please upload an image file.')
    return redirect(url_for('index'))

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Handle base64 image prediction (for paste functionality)"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_type = data.get('model_type', 'cnn')
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/...;base64, prefix
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        results, predicted_class, reliability_flag = predict_image(image_tensor, model_type)
        if results is None:
            return jsonify({'error': f'{model_type.upper()} model not available'})
        
        return jsonify({
            'results': results,
            'predicted_class': predicted_class,
            'model_type': model_type.upper(),
            'reliability_flag': reliability_flag
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

# Initialize models when the module is imported
def initialize_app():
    """Initialize the Flask app with models and directories"""
    global cnn_model, dnn_model, CLASS_NAMES
    
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load class names and models
    load_class_names()
    load_models()
    
    print(f"Using device: {device}")
    print(f"Class names: {CLASS_NAMES}")
    print(f"CNN model loaded: {cnn_model is not None}")
    print(f"DNN model loaded: {dnn_model is not None}")

# Initialize when module is imported
initialize_app()

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
