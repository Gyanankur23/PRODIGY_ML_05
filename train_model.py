import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_CLASSES = 101  # Food-101 has 101 classes

def create_data_generators(data_dir):
    """Create train and validation data generators"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def build_model(num_classes):
    """Build transfer learning model using MobileNetV2"""
    # Load pre-trained MobileNetV2 without top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(data_dir):
    """Train the food recognition model"""
    print("Creating data generators...")
    train_generator, validation_generator = create_data_generators(data_dir)
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_generator.class_indices}")
    
    # Save class indices
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    print("Building model...")
    model = build_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model summary:")
    model.summary()
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save the model
    model.save('food_recognition_model.h5')
    print("Model saved as 'food_recognition_model.h5'")
    
    return model, history

def train_with_sample_data():
    """Train with a simple sample dataset structure"""
    # Create a simple dataset structure for demonstration
    sample_data_dir = 'sample_food_data'
    
    if not os.path.exists(sample_data_dir):
        print(f"Creating sample data directory: {sample_data_dir}")
        os.makedirs(sample_data_dir, exist_ok=True)
        
        # Create subdirectories for a few food categories
        categories = ['pizza', 'burger', 'salad', 'pasta', 'rice']
        for category in categories:
            os.makedirs(os.path.join(sample_data_dir, category), exist_ok=True)
        
        print(f"Created directories for: {categories}")
        print(f"Please add images to each category directory in {sample_data_dir}")
        print("Each category should have at least 10 images for training.")
        return None
    
    return train_model(sample_data_dir)

if __name__ == '__main__':
    # Check for Indian Food Images directory first
    data_dirs = ['Indian Food Images', 'food-101/images', 'sample_food_data']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"Training with data from: {data_dir}")
            model, history = train_model(data_dir)
            break
    else:
        print("No data directory found. Creating sample structure...")
        model, history = train_with_sample_data()
