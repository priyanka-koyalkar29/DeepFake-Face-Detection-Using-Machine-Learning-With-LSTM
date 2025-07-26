# DeepFake-Face-Detection-Using-Machine-Learning-With-LSTM
A machine learning project for detecting deepfake videos using Xception + LSTM.
# DeepFake Face Detection using Machine Learning (Xception + LSTM)

A deep learning web app to detect deepfake faces using a hybrid model combining Xception CNN with LSTM. Built using Python, Flask, and OpenCV.

---

## 🔍 Features

- Real-time face detection using MTCNN
- DeepFake classification using Xception + LSTM model
- User-friendly web interface (Flask)
- SQLite database to store results

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repo

```bash
git clone https://github.com/priyanka-koyalkar29/DeepFake_Face_Detection_Using_LSTM.git
cd DeepFake_Face_Detection_Using_LSTM


2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Prepare the Pre-trained Model
🚨 Important: The model.h5 file (the trained deepfake detection model) is not included in the repository due to size limits.

You have two options:

Option A: Train the model yourself using train_model.py

Option B: Once trained, save the model as model.h5 and place it in the root folder.
# In train_model.py (example)
model.save("model.h5")

5. Run the Flask App
python app.py

Then open your browser and go to:
👉 http://127.0.0.1:5000

🧠 Model Architecture
Preprocessing: MTCNN for face detection from images

Model: Xception CNN (for spatial features) + LSTM (for temporal features)

Output: Binary classification – Real or DeepFake

📁 Project Structure
├── app.py                 # Main Flask Web App
├── model.py              # Load pre-trained model
├── train_model.py        # Training script
├── preprocess_data.py    # Face detection using MTCNN
├── deepfake_model.py     # Hybrid CNN + LSTM model definition
├── database.py           # SQLite database interface
├── deepfake_detection.db # SQLite DB file
├── templates/            # HTML templates
│   └── index.html        # Main UI page
├── static/               # Static assets (CSS, JS, images)
├── requirements.txt      # Project dependencies
├── README.md             # You're here!

🧪 Training the Model (Optional)
python train_model.py

📚 References
Xception CNN Paper

MTCNN GitHub Repo

🙋‍♀️ Author
Priyanka Koyalkar
Gold Medalist in Statistics | AI/ML Developer | MCA Graduate
 https://www.linkedin.com/in/k-p-priyanka 


