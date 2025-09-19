# CNN Image Classifier

A convolutional neural network implementation for image classification on CINIC-10 dataset. This model uses convolutional layers to automatically learn spatial features from 32x32 RGB images across 10 different classes.

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the model
python src/cnn_model.py
```

## Project Structure

```
CNN/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    └── cnn_model.py    # Main CNN implementation
```

## Model Architecture

The CNN consists of three main components:

### Convolutional Base
- **Conv Layer 1**: 3→32 channels, 3x3 kernel, ReLU activation, 2x2 max pooling
- **Conv Layer 2**: 32→64 channels, 3x3 kernel, ReLU activation, 2x2 max pooling  
- **Conv Layer 3**: 64→64 channels, 3x3 kernel, ReLU activation

### Dense Classifier
- **Flatten Layer**: Converts 8x8x64 feature maps to 4096-dimensional vector
- **Hidden Layer**: 4096→64 neurons, ReLU activation
- **Output Layer**: 64→10 neurons (one per class)

### Total Parameters
Approximately 262,000 trainable parameters

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | Default (0.001) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Image Size | 32x32x3 |
| Normalization | Mean: (0.478, 0.472, 0.430), Std: (0.242, 0.238, 0.258) |

## Data Preprocessing

- Images resized to 32x32 pixels
- Normalized using CINIC-10 dataset statistics
- Standard tensor conversion for PyTorch compatibility

## Performance Visualization

The model includes comprehensive visualization features:
- Training/validation loss and accuracy curves
- Per-class accuracy breakdown
- Feature map visualization from convolutional layers
- Sample images from each class

## Hardware Requirements

- CPU: Any modern processor (GPU/MPS acceleration supported)
- RAM: Minimum 4GB recommended
- Storage: ~500MB for dataset and model files

## Expected Performance

Typical results after 10 epochs:
- Training Accuracy: 70-80%
- Validation Accuracy: 65-75%
- Test Accuracy: 60-70%

Performance varies based on hardware and random initialization.
