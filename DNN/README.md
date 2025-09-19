# Deep Neural Network (MLP) Image Classifier

A fully-connected deep neural network implementation for image classification on CINIC-10 dataset. This model flattens input images and uses dense layers to learn patterns across all pixel values simultaneously.

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the model
python src/dnn_model.py
```

## Project Structure

```
DNN/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    └── dnn_model.py    # Main DNN implementation
```

## Model Architecture

The DNN uses a multi-layer perceptron (MLP) architecture:

### Input Processing
- **Flatten Layer**: Converts 32x32x3 images to 3072-dimensional vectors
- **Input Dimension**: 3072 features per image

### Hidden Layers
- **Layer 1**: 3072→512 neurons, ReLU activation
- **Layer 2**: 512→128 neurons, ReLU activation
- **Output Layer**: 128→10 neurons (one per class)

### Total Parameters
Approximately 1.6 million trainable parameters

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | Default (0.001) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Input Size | 3072 (32×32×3 flattened) |
| Normalization | Mean: (0.478, 0.472, 0.430), Std: (0.242, 0.238, 0.258) |

## Data Preprocessing

- Images resized to 32x32 pixels
- Flattened to 1D vectors (3072 dimensions)
- Normalized using CINIC-10 dataset statistics
- Standard tensor conversion for PyTorch compatibility

## Performance Visualization

The model includes comprehensive analysis features:
- Training/validation loss and accuracy curves
- Per-class accuracy breakdown
- Class distribution visualization across train/validation/test sets
- Sample images from each class with pie chart distribution

## Hardware Requirements

- CPU: Any modern processor (GPU/MPS acceleration supported)
- RAM: Minimum 6GB recommended (larger model than CNN)
- Storage: ~500MB for dataset and model files

## Expected Performance

Typical results after 10 epochs:
- Training Accuracy: 50-60%
- Validation Accuracy: 45-55%
- Test Accuracy: 40-50%

Note: MLPs typically underperform CNNs on image tasks as they don't preserve spatial relationships.

## Model Comparison

| Aspect | DNN (MLP) | CNN |
|--------|-----------|-----|
| Parameters | ~1.6M | ~262K |
| Spatial Awareness | None | High |
| Training Speed | Fast | Medium |
| Memory Usage | High | Medium |
| Image Performance | Lower | Higher |
