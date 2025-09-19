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
def visualize_data_distribution():
    """Visualize class distribution across datasets"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
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
        if i < 10:  # Show first 10 classes
            ax1.subplot = plt.subplot(2, 5, i+1) if i < 10 else None
            if i < 10:
                plt.subplot(2, 5, i+1)
                plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)  # Denormalize
                plt.title(class_names[i])
                plt.axis('off')
    
    plt.suptitle('Sample Images from Each Class')
    plt.tight_layout()
    plt.show()
    
    # Class distribution
    train_counts = [0] * len(class_names)
    val_counts = [0] * len(class_names)
    test_counts = [0] * len(class_names)
    
    for _, label in train_dataset:
        train_counts[label] += 1
    for _, label in val_dataset:
        val_counts[label] += 1
    for _, label in test_dataset:
        test_counts[label] += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution bar chart
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart for total distribution
    total_counts = [t + v + te for t, v, te in zip(train_counts, val_counts, test_counts)]
    ax2.pie(total_counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Overall Class Distribution')
    
    plt.tight_layout()
    plt.show()

visualize_data_distribution()

# Define DNN model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32*32*3, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
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

print("Training DNN...")
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
def plot_training_performance():
    """Plot training and validation metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('DNN Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('DNN Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-class accuracy
    class_names = train_dataset.classes
    class_accs = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                  for i in range(len(class_names))]
    
    ax3.bar(class_names, class_accs, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Per-Class Test Accuracy (DNN)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Model comparison metrics
    metrics = ['Train Acc', 'Val Acc', 'Test Acc']
    values = [train_accuracies[-1], val_accuracies[-1], test_accuracy]
    colors = ['blue', 'orange', 'green']
    
    ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Final Model Performance (DNN)')
    ax4.set_ylim(0, 100)
    
    for i, v in enumerate(values):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_training_performance()

# Save model
torch.save(model.state_dict(), 'dnn_model.pth')
print("Model saved as dnn_model.pth")

# Also save to demo/models folder for Flask app
demo_models_path = '../../demo/models/dnn_model.pth'
try:
    torch.save(model.state_dict(), demo_models_path)
    print(f"Model also saved to {demo_models_path} for Flask demo")
except Exception as e:
    print(f"Note: Could not save to demo folder: {e}")
