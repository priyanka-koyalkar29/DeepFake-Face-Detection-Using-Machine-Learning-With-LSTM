from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import from our modules
from database import register_user, validate_user, save_detection, get_user_detections
from model import preprocess_image

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
MODEL_PATH = os.path.join('models', 'trained_model.h5')

# We'll load the model when needed to prevent loading during testing or if model doesn't exist yet
model = None

def get_model():
    """Load the model if it exists, otherwise return None"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            flash("Model not found. Please train the model first.", "error")
    return model

def allowed_file(filename):
    """Check if the file is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """Home page route"""
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if not username or not password or not email:
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if register_user(username, password, email):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = validate_user(username, password)
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout route"""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    """User dashboard page"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    user_detections = get_user_detections(session['user_id'])
    return render_template('dashboard.html', detections=user_detections)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Deepfake detection page"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Check if file type is allowed
        if file and allowed_file(file.filename):
            # Generate a unique filename to prevent overwriting
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_path)
            
            # Get the model
            model = get_model()
            if model is None:
                return redirect(url_for('dashboard'))
            
            # Preprocess the image
            img = preprocess_image(file_path)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(img)[0][0]
            
            # Determine result
            result = "FAKE" if prediction > 0.5 else "REAL"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            confidence = float(confidence) * 100  # Convert to percentage
            
            # Save the detection to database
            save_detection(
                session['user_id'],
                os.path.join('uploads', unique_filename),  # Store relative path
                result,
                confidence
            )
            
            return render_template('result.html', 
                                  image_path=os.path.join('uploads', unique_filename),
                                  result=result,
                                  confidence=confidence)
        else:
            flash('File type not allowed. Please upload an image file (jpg, jpeg, png).', 'error')
            return redirect(request.url)
    
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
