import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, GlobalAveragePooling2D, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from mtcnn import MTCNN

class DeepFakeDetector:
    def __init__(self):
        self.face_detector = MTCNN()
        self.sequence_length = 20  # Number of frames to process in a sequence
        self.img_size = 224
        self.model = None
        self.model_path = "models/deepfake_lstm_model.h5"
        
    def build_model(self):
        """Build the LSTM-based deep fake detection model"""
        # Base model - Xception for feature extraction
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(self.img_size, self.img_size, 3))
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Create LSTM model architecture
        input_layer = Input(shape=(self.sequence_length, self.img_size, self.img_size, 3))
        
        # TimeDistributed applies the same operation to each time step
        x = TimeDistributed(base_model)(input_layer)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        
        # LSTM layers
        x = LSTM(2048, return_sequences=True)(x)
        x = Dropout(0.6)(x)
        x = LSTM(1024)(x)
        x = Dropout(0.6)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
        self.model = model
        return model
    
    def load_trained_model(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            return True
        return False
    
    def detect_and_crop_face(self, image):
        """Detect and crop the face from an image"""
        # Convert RGB to BGR if needed (for OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = image
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detect_faces(img_rgb)
        
        if not faces:
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add some margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_rgb.shape[1] - x, w + 2*margin)
        h = min(img_rgb.shape[0] - y, h + 2*margin)
        
        # Crop and resize
        face_crop = img_rgb[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (self.img_size, self.img_size))
        
        return face_crop
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Detect and crop face
        face = self.detect_and_crop_face(frame)
        
        if face is None:
            return None
        
        # Normalize
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def extract_frames(self, video_path, max_frames=20):
        """Extract frames from a video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step to evenly distribute frames
        step = max(1, total_frames // max_frames)
        
        frame_indices = []
        for i in range(0, min(total_frames, max_frames * step), step):
            frame_indices.append(i)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                processed_frame = self.preprocess_frame(frame)
                if processed_frame is not None:
                    frames.append(processed_frame)
        
        cap.release()
        
        # Pad or truncate to ensure we have exactly max_frames
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        elif len(frames) < max_frames:
            # Pad by duplicating the last frame
            last_frame = frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3))
            while len(frames) < max_frames:
                frames.append(last_frame)
        
        return np.array(frames)
    
    def predict_video(self, video_path):
        """Predict if a video contains deep fakes"""
        if self.model is None:
            if not self.load_trained_model():
                self.build_model()
                print("No trained model found. Please train the model first.")
                return None
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            return {"error": "No faces detected in the video"}
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        
        # Make prediction
        prediction = self.model.predict(frames)[0][0]
        
        result = {
            "prediction": "fake" if prediction > 0.5 else "real",
            "confidence": float(prediction) if prediction > 0.5 else float(1 - prediction)
        }
        
        return result
    
    def predict_image(self, image_path):
        """Predict if an image contains deep fakes"""
        if self.model is None:
            if not self.load_trained_model():
                self.build_model()
                print("No trained model found. Please train the model first.")
                return None
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        processed_image = self.preprocess_frame(image)
        
        if processed_image is None:
            return {"error": "No face detected in the image"}
        
        # Create a sequence by duplicating the image
        frames = np.array([processed_image] * self.sequence_length)
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        
        # Make prediction
        prediction = self.model.predict(frames)[0][0]
        
        result = {
            "prediction": "fake" if prediction > 0.5 else "real",
            "confidence": float(prediction) if prediction > 0.5 else float(1 - prediction)
        }
        
        return result
    
    def train(self, train_dir, validation_dir, epochs=20, batch_size=16):
        """Train the model on the dataset"""
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_generator = self.create_data_generator(train_dir, batch_size)
        validation_generator = self.create_data_generator(validation_dir, batch_size)
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def create_data_generator(self, data_dir, batch_size):
        """Create a data generator for training"""
        # This is a placeholder function - in a real implementation,
        # you would create a custom data generator that:
        # 1. Reads videos from data_dir
        # 2. Extracts frames
        # 3. Preprocesses frames
        # 4. Yields batches of sequences and labels
        
        real_dir = os.path.join(data_dir, "Real")
        fake_dir = os.path.join(data_dir, "fake")
        
        # This is a simplified generator that would need to be expanded
        def generator():
            while True:
                # Generate a batch of real samples
                real_samples = []
                for i in range(batch_size // 2):
                    # TODO: Implement loading a random real video and extracting frames
                    pass
                
                # Generate a batch of fake samples
                fake_samples = []
                for i in range(batch_size // 2):
                    # TODO: Implement loading a random fake video and extracting frames
                    pass
                
                # Combine and yield
                X = np.array(real_samples + fake_samples)
                y = np.array([0] * (batch_size // 2) + [1] * (batch_size // 2))
                
                yield X, y
        
        return generator()
