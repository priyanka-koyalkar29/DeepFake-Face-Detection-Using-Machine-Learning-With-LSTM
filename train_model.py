import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import argparse
import random
from deepfake_model import DeepFakeDetector
import json
import matplotlib.pyplot as plt

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

class DeepFakeSequence(Sequence):
    """Data generator for training the deepfake detection model"""
    
    def __init__(self, data_dir, batch_size=4, seq_length=20, img_size=224, is_training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.img_size = img_size
        self.is_training = is_training
        
        # Get real and fake sequence directories
        self.real_dirs = self._get_sequence_dirs(os.path.join(data_dir, "Real"))
        self.fake_dirs = self._get_sequence_dirs(os.path.join(data_dir, "fake"))
        
        print(f"Found {len(self.real_dirs)} real sequences and {len(self.fake_dirs)} fake sequences")
        
        # Create list of (sequence_dir, label) pairs
        self.sequences = [(seq_dir, 0) for seq_dir in self.real_dirs] + [(seq_dir, 1) for seq_dir in self.fake_dirs]
        
        # Shuffle sequences
        random.shuffle(self.sequences)
    
    def _get_sequence_dirs(self, category_dir):
        """Get all sequence directories in a category directory"""
        if not os.path.exists(category_dir):
            return []
            
        return [os.path.join(category_dir, d) for d in os.listdir(category_dir) 
                if os.path.isdir(os.path.join(category_dir, d))]
    
    def __len__(self):
        """Return the number of batches"""
        return len(self.sequences) // self.batch_size
    
    def __getitem__(self, idx):
        """Get a batch"""
        batch_sequences = self.sequences[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_X = np.zeros((self.batch_size, self.seq_length, self.img_size, self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros(self.batch_size, dtype=np.float32)
        
        for i, (seq_dir, label) in enumerate(batch_sequences):
            # Get all frame files
            frame_files = [f for f in os.listdir(seq_dir) if f.startswith("frame_") and f.endswith(".jpg")]
            frame_files.sort()
            
            # If we don't have enough frames, duplicate the last frames
            if len(frame_files) < self.seq_length:
                frame_files = frame_files + [frame_files[-1]] * (self.seq_length - len(frame_files))
            
            # If we have too many frames, select evenly spaced frames
            if len(frame_files) > self.seq_length:
                indices = np.linspace(0, len(frame_files) - 1, self.seq_length, dtype=int)
                frame_files = [frame_files[i] for i in indices]
            
            # Load and preprocess frames
            for j, frame_file in enumerate(frame_files[:self.seq_length]):
                frame_path = os.path.join(seq_dir, frame_file)
                img = load_img(frame_path, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img) / 255.0
                batch_X[i, j] = img_array
                
            batch_y[i] = label
            
        return batch_X, batch_y
    
    def on_epoch_end(self):
        """Shuffle sequences at the end of each epoch"""
        if self.is_training:
            random.shuffle(self.sequences)

def plot_training_history(history, output_dir):
    """Plot training history and save figures"""
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))

def train_model(args):
    """Train the deepfake detection model"""
    print("Creating data generators...")
    train_generator = DeepFakeSequence(
        args.train_dir,
        batch_size=args.batch_size,
        seq_length=args.sequence_length,
        img_size=args.img_size,
        is_training=True
    )
    
    val_generator = DeepFakeSequence(
        args.val_dir,
        batch_size=args.batch_size,
        seq_length=args.sequence_length,
        img_size=args.img_size,
        is_training=False
    )
    
    print("Creating model...")
    detector = DeepFakeDetector()
    detector.sequence_length = args.sequence_length
    detector.img_size = args.img_size
    detector.model_path = args.model_path
    
    # Build or load model
    if os.path.exists(args.model_path) and not args.fresh_start:
        print(f"Loading existing model from {args.model_path}")
        detector.load_trained_model()
    else:
        print("Building new model")
        detector.build_model()
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    history = detector.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.workers > 1
    )
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        history_dict = {key: [float(x) for x in val] for key, val in history.history.items()}
        json.dump(history_dict, f, indent=4)
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    print(f"Training complete. Model saved to {args.model_path}")
    print(f"Training history saved to {history_path}")
    print(f"Training plots saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory containing processed training data")
    parser.add_argument("--val_dir", type=str, required=True, help="Directory containing processed validation data")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model and results")
    parser.add_argument("--model_path", type=str, default="models/deepfake_lstm_model.h5", help="Path to save model")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--sequence_length", type=int, default=20, help="Number of frames per sequence")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--fresh_start", action="store_true", help="Start training from scratch")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train_model(args)

if __name__ == "__main__":
    main()
