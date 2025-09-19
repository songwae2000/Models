"""
Model definitions that match the exact architectures used in training.
These must be identical to the models defined in CNN/src/cnn_model.py and DNN/src/dnn_model.py
"""
import torch
import torch.nn as nn

# Create models exactly as they are in the training scripts
# This ensures the state_dict keys match perfectly

def create_cnn_model():
    """Create CNN model exactly as in CNN/src/cnn_model.py"""
    return nn.Sequential(
        # Convolutional base
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        
        # Dense layers
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def create_dnn_model():
    """Create DNN model exactly as in DNN/src/dnn_model.py"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# For backward compatibility
CNNModel = create_cnn_model
DNNModel = create_dnn_model
