import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from model import create_cnn_lstm_model, create_simple_cnn_model, create_train_val_sets

def train_model(real_dir, fake_dir, model_path, epochs=20, batch_size=32, img_size=128, use_simple=False):
    """
    Train the deepfake detection model
    
    Args:
        real_dir: Directory containing real face images
        fake_dir: Directory containing fake/deepfake images
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Target size for images
        use_simple: Whether to use the simple CNN model instead of CNN-LSTM
    """
    # Create the directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load and prepare the dataset
    print("Loading and preparing dataset...")
    X_train, X_val, y_train, y_val = create_train_val_sets(
        real_dir, fake_dir, target_size=(img_size, img_size), test_size=0.2
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    
    # Create the model
    print("Creating model...")
    if use_simple:
        model = create_simple_cnn_model(input_shape=(img_size, img_size, 3))
    else:
        model = create_cnn_lstm_model(input_shape=(img_size, img_size, 3))
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_path), 'training_history.png'))
    
    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation loss: {loss:.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deepfake detection model")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real face images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory with fake face images")
    parser.add_argument("--model_path", type=str, default="models/trained_model.h5", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=128, help="Target image size")
    parser.add_argument("--use_simple", action="store_true", help="Use simple CNN model instead of CNN-LSTM")
    
    args = parser.parse_args()
    
    train_model(
        args.real_dir,
        args.fake_dir,
        args.model_path,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.use_simple
    )
