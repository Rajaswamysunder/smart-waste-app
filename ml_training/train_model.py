#!/usr/bin/env python3
"""
Waste Classifier Model Training Script
======================================
Trains a MobileNetV2-based classifier for waste detection.

Categories:
- Organic (food waste, leaves, compost)
- Recyclable (plastic, paper, glass, metal)
- Hazardous (batteries, chemicals)
- E-Waste (electronics, cables)
- General (mixed waste)

Usage:
    python train_model.py [--epochs 50] [--batch_size 32]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau,
    TensorBoard
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Constants
IMG_SIZE = 224
CLASSES = ['organic', 'recyclable', 'hazardous', 'ewaste', 'general']
NUM_CLASSES = len(CLASSES)

def create_dataset_structure():
    """Create dataset directory structure if it doesn't exist."""
    dataset_dir = Path('dataset')
    
    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a placeholder file
        readme = class_dir / 'README.txt'
        if not readme.exists():
            readme.write_text(f"""
Add {class_name} waste images here.

Examples of {class_name} waste:
{get_class_examples(class_name)}

Recommended:
- At least 500 images
- Various lighting conditions
- Different angles
- Mix of backgrounds
""")
    
    print(f"✅ Dataset structure created at: {dataset_dir.absolute()}")
    return dataset_dir

def get_class_examples(class_name):
    """Return examples for each waste class."""
    examples = {
        'organic': '- Food scraps\n- Fruit peels\n- Vegetable waste\n- Leaves\n- Garden waste\n- Coffee grounds',
        'recyclable': '- Plastic bottles\n- Aluminum cans\n- Paper/cardboard\n- Glass bottles\n- Metal containers',
        'hazardous': '- Batteries\n- Paint cans\n- Chemical containers\n- Pesticides\n- Medical waste',
        'ewaste': '- Old phones\n- Cables/wires\n- Computer parts\n- Keyboards\n- Broken electronics',
        'general': '- Mixed waste\n- Non-recyclable plastic\n- Styrofoam\n- Diapers\n- General trash'
    }
    return examples.get(class_name, '')

def count_images(dataset_dir):
    """Count images in each class directory."""
    counts = {}
    total = 0
    
    print("\n📊 Dataset Statistics:")
    print("-" * 40)
    
    for class_name in CLASSES:
        class_dir = Path(dataset_dir) / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.jpeg')) + \
                    list(class_dir.glob('*.png'))
            count = len(images)
            counts[class_name] = count
            total += count
            status = "✅" if count >= 100 else "⚠️"
            print(f"{status} {class_name:15} : {count:5} images")
        else:
            counts[class_name] = 0
            print(f"❌ {class_name:15} : 0 images (folder missing)")
    
    print("-" * 40)
    print(f"   {'TOTAL':15} : {total:5} images")
    
    return counts, total

def create_model(num_classes, fine_tune_layers=50):
    """Create MobileNetV2-based transfer learning model."""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers (except last few for fine-tuning)
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Create model
    model = keras.Sequential([
        # Input preprocessing
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Data augmentation (only during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        
        # Base model
        base_model,
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_data_generators(dataset_dir, batch_size, validation_split=0.2):
    """Create training and validation data generators."""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    print(f"\n📂 Loading images from: {dataset_dir}")
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        subset='validation',
        shuffle=False
    )
    
    print(f"\n✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    
    return train_generator, val_generator

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training history saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Confusion matrix saved to: {save_path}")
    plt.close()

def train(epochs=50, batch_size=32, dataset_dir='dataset'):
    """Main training function."""
    
    print("\n" + "="*60)
    print("🗑️  WASTE CLASSIFIER MODEL TRAINING")
    print("="*60)
    
    # Create dataset structure
    dataset_path = create_dataset_structure()
    
    # Count images
    counts, total = count_images(dataset_path)
    
    if total < 50:
        print("\n⚠️  WARNING: Not enough images for training!")
        print("Please add at least 100 images per category.")
        print("\nYou can download datasets from:")
        print("1. TrashNet: https://github.com/garythung/trashnet")
        print("2. Kaggle: https://www.kaggle.com/datasets/techsash/waste-classification-data")
        
        # Create sample dataset for testing
        create_sample = input("\nCreate sample synthetic dataset for testing? (y/n): ")
        if create_sample.lower() == 'y':
            create_synthetic_dataset(dataset_path)
            counts, total = count_images(dataset_path)
        else:
            return
    
    # Create output directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(dataset_path, batch_size)
    
    # Create model
    print("\n🔧 Creating model...")
    model = create_model(NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/waste_classifier_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Classes: {CLASSES}")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/waste_classifier_final.keras')
    print("\n✅ Model saved to: models/waste_classifier_final.keras")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    print("\n📊 Evaluating model...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Save labels
    with open('models/waste_labels.txt', 'w') as f:
        for label in CLASSES:
            f.write(f"{label}\n")
    print("✅ Labels saved to: models/waste_labels.txt")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print("1. Run: python convert_to_tflite.py")
    print("2. Copy model to Flutter app")
    
    return model, history

def create_synthetic_dataset(dataset_dir, samples_per_class=100):
    """Create synthetic dataset for testing (colored rectangles)."""
    from PIL import Image
    import random
    
    print("\n🔧 Creating synthetic dataset for testing...")
    
    # Color profiles for each class
    color_profiles = {
        'organic': [(139, 90, 43), (34, 139, 34), (154, 205, 50)],  # Browns, greens
        'recyclable': [(70, 130, 180), (192, 192, 192), (255, 255, 224)],  # Blues, silvers
        'hazardous': [(255, 0, 0), (255, 165, 0), (255, 255, 0)],  # Reds, oranges, yellows
        'ewaste': [(47, 79, 79), (0, 0, 0), (105, 105, 105)],  # Dark grays, blacks
        'general': [(128, 128, 128), (169, 169, 169), (211, 211, 211)]  # Grays
    }
    
    for class_name, colors in color_profiles.items():
        class_dir = Path(dataset_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            # Create image with random noise and class colors
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            pixels = img.load()
            
            base_color = random.choice(colors)
            
            for x in range(IMG_SIZE):
                for y in range(IMG_SIZE):
                    # Add noise
                    noise = random.randint(-30, 30)
                    r = max(0, min(255, base_color[0] + noise + random.randint(-20, 20)))
                    g = max(0, min(255, base_color[1] + noise + random.randint(-20, 20)))
                    b = max(0, min(255, base_color[2] + noise + random.randint(-20, 20)))
                    pixels[x, y] = (r, g, b)
            
            img.save(class_dir / f'synthetic_{i:04d}.jpg', 'JPEG')
        
        print(f"  ✅ Created {samples_per_class} images for {class_name}")
    
    print("✅ Synthetic dataset created!")

def main():
    parser = argparse.ArgumentParser(description='Train Waste Classifier Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset directory')
    
    args = parser.parse_args()
    
    # Check TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_dir=args.dataset
    )

if __name__ == '__main__':
    main()
