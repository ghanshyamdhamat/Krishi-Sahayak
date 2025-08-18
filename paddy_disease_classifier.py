import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize,
    RandomHorizontalFlip, RandomRotation, RandomErasing,
    RandomAdjustSharpness, ColorJitter
)
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from tqdm import tqdm
from collections import Counter
import random

# --- Label Mapping ---
LABEL_MAP = {
    'normal': 0,
    'bacterial_leaf_blight': 1,
    'bacterial_leaf_streak': 2,
    'bacterial_panicle_blight': 3,
    'blast': 4,
    'brown_spot': 5,
    'dead_heart': 6,
    'downy_mildew': 7,
    'hispa': 8,
    'tungro': 9
}

# --- Transformations ---
train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(20),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.25, scale=(0.02, 0.2))
])

val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset Class (Unchanged) ---
class PaddyDataset(Dataset):
    def __init__(self, img_list, labels, transform=None):
        self.img_list = img_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Classifier Class (Unchanged) ---
class PaddyDiseaseClassifier(nn.Module):
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, val_loader, device='cuda'):
        super(PaddyDiseaseClassifier, self).__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _train_epoch(self):
        self.model.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        return running_loss / total_samples, correct_predictions / total_samples

    def _validate_epoch(self):
        self.model.eval()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        return running_loss / total_samples, correct_predictions / total_samples

    def train(self, epochs, save_path='best_model.pth'):
        best_val_acc = 0.0
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ New best model saved to {save_path} with accuracy: {best_val_acc:.4f}")
        print("Training finished.")
        return self.history

# --- Plotting Function (Unchanged) ---
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy History'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True)
    plt.show()

# --- *** CORRECTED JSON CREATION AND SPLITTING *** ---
def create_json(image_dir, output_json, val_split=0.2):
    """Creates a JSON file by correctly splitting data into train and validation sets."""
    all_data = []
    for label_name, label_idx in LABEL_MAP.items():
        class_dir = os.path.join(image_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            all_data.append({'image': img_path, 'label': label_idx})
            
    # Shuffle and split the data
    random.shuffle(all_data)
    split_idx = int(len(all_data) * (1 - val_split))
    
    data = {
        'train': all_data[:split_idx],
        'val': all_data[split_idx:]
    }
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON file created at {output_json} with {len(data['train'])} train and {len(data['val'])} val samples.")


if __name__ == '__main__':
    # --- Configuration ---
    IMAGE_DIR = '/mnt/bb586fde-943d-4653-af27-224147bfba7e/Capital_One/paddy-disease-classification/train_images'
    JSON_PATH = os.path.join(IMAGE_DIR, 'splits.json')
    
    # Create JSON file if it doesn't exist
    if not os.path.exists(JSON_PATH):
        create_json(IMAGE_DIR, JSON_PATH)
        
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 100 # 500 is very long, start with a smaller number
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Load Data from JSON ---
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # *** CORRECTED PATH HANDLING ***
    # The paths in the JSON are already absolute, no need to join with IMAGE_DIR again.
    train_img_paths = [item['image'] for item in data['train']]
    train_labels = [item['label'] for item in data['train']]
    val_img_paths = [item['image'] for item in data['val']]
    val_labels = [item['label'] for item in data['val']]
    
    train_dataset = PaddyDataset(train_img_paths, train_labels, transform=train_transform)
    val_dataset = PaddyDataset(val_img_paths, val_labels, transform=val_transform)
    
    # --- *** IMPLEMENT WEIGHTED SAMPLER FOR IMBALANCED DATASET *** ---
    class_counts = Counter(train_labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Use the sampler in the train_loader, shuffle must be False
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- Initialize Model, Optimizer, and Criterion ---
    model = models.resnet34(weights='IMAGENET1K_V1')
    
    # Unfreeze more layers for better feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last two blocks (layer3 and layer4)
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace the final classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )

    # Optimize all trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # --- Create Classifier and Start Training ---
    classifier = PaddyDiseaseClassifier(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE
    )
    
    history = classifier.train(epochs=EPOCHS, save_path='paddy_disease_best_model.pth')
    
    plot_history(history)