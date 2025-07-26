import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import concurrent.futures
import argparse

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

class DatasetPreprocessor:
    def __init__(self, input_dir, output_dir, img_size=224, sequence_length=20):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.face_detector = MTCNN()
        
        # Create output directories
        self.processed_real_dir = os.path.join(output_dir, "processed", "Real")
        self.processed_fake_dir = os.path.join(output_dir, "processed", "fake")
        create_directory(self.processed_real_dir)
        create_directory(self.processed_fake_dir)
        
    def detect_and_crop_face(self, image):
        """Detect and crop face from an image"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            
        # Detect faces
        faces = self.face_detector.detect_faces(img_rgb)
        
        if not faces:
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_rgb.shape[1] - x, w + 2*margin)
        h = min(img_rgb.shape[0] - y, h + 2*margin)
        
        # Crop and resize
        face_crop = img_rgb[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (self.img_size, self.img_size))
        
        return cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    
    def process_video(self, video_path, output_dir, category):
        """Process a video into a sequence of face frames"""
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_sequence_dir = os.path.join(output_dir, base_name)
        create_directory(output_sequence_dir)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        step = max(1, total_frames // self.sequence_length)
        
        # Create numpy array for storing processed frames
        processed_frames = []
        
        # Process frames
        frame_idx = 0
        for i in range(0, min(total_frames, self.sequence_length * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Detect and crop face
            face = self.detect_and_crop_face(frame)
            
            if face is not None:
                # Save face image
                output_path = os.path.join(output_sequence_dir, f"frame_{frame_idx:03d}.jpg")
                cv2.imwrite(output_path, face)
                processed_frames.append(face)
                frame_idx += 1
        
        cap.release()
        
        # If we couldn't extract enough faces, remove the directory
        if len(processed_frames) < self.sequence_length // 2:
            os.rmdir(output_sequence_dir)
            return False
            
        # Save metadata
        with open(os.path.join(output_sequence_dir, "metadata.txt"), "w") as f:
            f.write(f"original_video: {video_path}\n")
            f.write(f"category: {category}\n")
            f.write(f"frames_extracted: {len(processed_frames)}\n")
            
        return True
    
    def process_image(self, image_path, output_dir, category):
        """Process a single image to detect and save the face"""
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")
        
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            return False
            
        # Detect and crop face
        face = self.detect_and_crop_face(image)
        
        if face is None:
            return False
            
        # Save face image
        cv2.imwrite(output_path, face)
        
        return True
    
    def process_dataset(self):
        """Process the entire dataset"""
        # Process real videos/images
        real_dir = os.path.join(self.input_dir, "Real")
        if os.path.exists(real_dir):
            print(f"Processing real videos/images from {real_dir}")
            self._process_directory(real_dir, self.processed_real_dir, "real")
            
        # Process fake videos/images
        fake_dir = os.path.join(self.input_dir, "fake")
        if os.path.exists(fake_dir):
            print(f"Processing fake videos/images from {fake_dir}")
            self._process_directory(fake_dir, self.processed_fake_dir, "fake")
    
    def _process_directory(self, input_dir, output_dir, category):
        """Process all files in a directory"""
        # Get all files
        files = []
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                files.append(os.path.join(root, filename))
                
        print(f"Found {len(files)} files in {input_dir}")
        
        # Process files with multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Create progress bar
            pbar = tqdm(total=len(files), desc=f"Processing {category} files")
            
            # Submit processing tasks
            future_to_file = {}
            for file_path in files:
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    future = executor.submit(self.process_video, file_path, output_dir, category)
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    future = executor.submit(self.process_image, file_path, output_dir, category)
                else:
                    continue
                    
                future_to_file[future] = file_path
                
            # Process results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"Failed to process {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                finally:
                    pbar.update(1)
                    
            pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Preprocess deepfake dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing Real and fake folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--img_size", type=int, default=224, help="Size of output face images")
    parser.add_argument("--sequence_length", type=int, default=20, help="Number of frames to extract per video")
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(
        args.input_dir,
        args.output_dir,
        args.img_size,
        args.sequence_length
    )
    
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()
