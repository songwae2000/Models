import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.478, 0.472, 0.430), (0.242, 0.238, 0.258)) 
])

# Load datasets
data_dir = '/Users/cajetan/Documents/Deep_Lerning/Prosit_1/archive'
train_dataset = datasets.ImageFolder(f'{data_dir}/train', transform=transform)
val_dataset = datasets.ImageFolder(f'{data_dir}/valid', transform=transform)
test_dataset = datasets.ImageFolder(f'{data_dir}/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Dataset sizes:")
print(f"  Training: {len(train_dataset):,} samples")
print(f"  Validation: {len(val_dataset):,} samples") 
print(f"  Test: {len(test_dataset):,} samples")
print(f"  Classes: {train_dataset.classes}")

# Data Visualization
def visualize_cnn_data():
    """Visualize data for CNN training"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    
    # Sample images from each class
    class_names = train_dataset.classes
    sample_images = []
    for class_idx in range(len(class_names)):
        for img, label in train_dataset:
            if label == class_idx:
                sample_images.append((img, label))
                break
    
    # Plot sample images
    for i, (img, label) in enumerate(sample_images):
        ax = axes[i // 5, i % 5]
        # Denormalize image for display
        img_display = img * torch.tensor([0.242, 0.238, 0.258]).view(3, 1, 1) + torch.tensor([0.478, 0.472, 0.430]).view(3, 1, 1)
        img_display = torch.clamp(img_display, 0, 1)
        ax.imshow(img_display.permute(1, 2, 0))
        ax.set_title(class_names[i])
        ax.axis('off')
    
    plt.suptitle('Sample Images from Each Class (CNN Input)')
    plt.tight_layout()
    plt.show()

visualize_cnn_data()

# Define CNN model
model = nn.Sequential(
    # Convolutional layers
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

# Compile model
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Train model
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")
model.to(device)

print("Training CNN...")
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(10):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            pred = output.argmax(dim=1)
            val_correct += pred.eq(target).sum().item()
            val_total += target.size(0)
    
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    # Store metrics
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Final test evaluation
model.eval()
test_correct = 0
test_total = 0
class_correct = [0] * len(train_dataset.classes)
class_total = [0] * len(train_dataset.classes)

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        
        test_correct += pred.eq(target).sum().item()
        test_total += target.size(0)
        
        # Per-class accuracy
        for i in range(target.size(0)):
            label = target[i].item()
            class_correct[label] += pred[i].eq(target[i]).item()
            class_total[label] += 1

test_accuracy = 100. * test_correct / test_total
print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')

# Performance Visualization
def plot_cnn_performance():
    """Plot comprehensive CNN training and performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('CNN Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, marker='o')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('CNN Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-class accuracy
    class_names = train_dataset.classes
    class_accs = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                  for i in range(len(class_names))]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax3.bar(class_names, class_accs, alpha=0.8, color=colors)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Per-Class Test Accuracy (CNN)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, class_accs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Model performance comparison
    metrics = ['Train Acc', 'Val Acc', 'Test Acc']
    values = [train_accuracies[-1], val_accuracies[-1], test_accuracy]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Final CNN Performance Summary')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    for bar, v in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Feature map visualization
def visualize_feature_maps():
    """Visualize CNN feature maps for a sample image"""
    model.eval()
    
    # Get a sample image
    sample_data, sample_target = next(iter(test_loader))
    sample_img = sample_data[0:1].to(device)  # Take first image
    
    # Extract features from each conv layer
    features = []
    x = sample_img
    
    for i, layer in enumerate(model):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            features.append(x)
    
    # Plot original image and feature maps
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    orig_img = sample_data[0]
    orig_img = orig_img * torch.tensor([0.242, 0.238, 0.258]).view(3, 1, 1) + torch.tensor([0.478, 0.472, 0.430]).view(3, 1, 1)
    orig_img = torch.clamp(orig_img, 0, 1)
    axes[0].imshow(orig_img.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Feature maps from each conv layer
    for i, feature_map in enumerate(features[:3]):
        # Average across all channels for visualization
        avg_feature = torch.mean(feature_map[0], dim=0).cpu().detach()
        im = axes[i+1].imshow(avg_feature, cmap='viridis')
        axes[i+1].set_title(f'Conv Layer {i+1} Features')
        axes[i+1].axis('off')
        plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
    
    plt.suptitle('CNN Feature Map Visualization')
    plt.tight_layout()
    plt.show()

plot_cnn_performance()
visualize_feature_maps()

# Save model
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved as cnn_model.pth")

# Also save to demo/models folder for Flask app
demo_models_path = '../../demo/models/cnn_model.pth'
try:
    torch.save(model.state_dict(), demo_models_path)
    print(f"Model also saved to {demo_models_path} for Flask demo")
except Exception as e:
    print(f"Note: Could not save to demo folder: {e}")
