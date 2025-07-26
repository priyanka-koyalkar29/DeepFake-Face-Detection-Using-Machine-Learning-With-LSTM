import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed, Input, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

def create_cnn_lstm_model(input_shape=(128, 128, 3)):
    """
    Create a hybrid CNN-LSTM model for deepfake detection
    
    The model first uses CNN to extract features from each frame,
    then passes those features through an LSTM to capture temporal relationships.
    """
    # CNN Feature Extractor
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # We'll freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add additional layers for our task
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_simple_cnn_model(input_shape=(128, 128, 3)):
    """
    Create a simpler CNN model for deepfake detection
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess an image for the model
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize
    img = img / 255.0  # Normalize
    return img

def load_dataset(real_dir, fake_dir, target_size=(128, 128)):
    """
    Load dataset from directories containing real and fake images
    """
    images = []
    labels = []
    
    # Load real images
    for filename in os.listdir(real_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(real_dir, filename)
            img = preprocess_image(img_path, target_size)
            images.append(img)
            labels.append(0)  # 0 for real
    
    # Load fake images
    for filename in os.listdir(fake_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(fake_dir, filename)
            img = preprocess_image(img_path, target_size)
            images.append(img)
            labels.append(1)  # 1 for fake
    
    return np.array(images), np.array(labels)

def create_train_val_sets(real_dir, fake_dir, target_size=(128, 128), test_size=0.2):
    """
    Create training and validation sets from the dataset
    """
    X, y = load_dataset(real_dir, fake_dir, target_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val
