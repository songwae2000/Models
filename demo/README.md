# Image Classifier Demo

A Flask web application for testing CNN and DNN models trained on CINIC-10 dataset. Upload images or paste from clipboard to compare model predictions.

## Quick Start

### Step 1: Train the Models
Before running the demo, train the models from their respective directories:

```bash
# Train CNN model (saves to both CNN/ and demo/models/)
cd CNN && python src/cnn_model.py

# Train DNN model (saves to both DNN/ and demo/models/)
cd ../DNN && python src/dnn_model.py
```

### Step 2: Run the Demo
```bash
# Navigate to demo directory
cd demo

# Create and activate virtual environment (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

**Note**: On macOS, you need to use a virtual environment due to system Python restrictions.

Open your browser and go to `http://localhost:5000`

## Features

- **File Upload**: Support for PNG, JPG, JPEG, GIF, BMP, WebP formats
- **Clipboard Paste**: Paste images directly from clipboard (Ctrl+V)
- **Drag & Drop**: Drag image files directly onto the upload area
- **Model Selection**: Choose between CNN and DNN models
- **Top-3 Predictions**: See confidence scores for top 3 classes
- **Model Comparison**: Side-by-side architecture comparison
- **Responsive Design**: Works on desktop and mobile devices

## Model Requirements

The application loads trained model files from:
- `demo/models/cnn_model.pth` - CNN model weights
- `demo/models/dnn_model.pth` - DNN model weights

These files are automatically created when you train the models using the CNN and DNN training scripts.

## CINIC-10 Classes

The models classify images into these 10 categories:
- airplane
- automobile  
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Technical Details

- **Framework**: Flask 2.3+
- **ML Backend**: PyTorch 2.0+
- **Frontend**: Bootstrap 5.1
- **Image Processing**: PIL/Pillow
- **Max Upload Size**: 16MB
- **Supported Devices**: CPU, CUDA, MPS (Apple Silicon)

## Usage Tips

1. **Best Results**: Use clear, well-lit images of the target objects
2. **Image Size**: Any size works (automatically resized to 32x32)
3. **Performance**: CNN typically outperforms DNN on image tasks
4. **Speed**: DNN predictions are faster but less accurate
