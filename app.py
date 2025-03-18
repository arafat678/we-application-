
# from flask import Flask, render_template, request, send_from_directory, url_for
# from fastai.vision.all import *
# import os
# import uuid
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import random
# from werkzeug.utils import secure_filename

# # Setup
# app = Flask(__name__)
# UPLOAD_FOLDER = 'predictions'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # Load Model
# MODEL_PATH = r"C:\\Users\\Administrator\\Desktop\\all fruits\\My all fruit_classifier1.pk1"
# if not os.path.exists(MODEL_PATH):
#     print(f"Error: Model file not found at {MODEL_PATH}")
#     exit(1)

# try:
#     learn = load_learner(MODEL_PATH)
#     print(f"Model loaded successfully: {learn.dls.vocab}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)

# # Helper Functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_dimensions(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
    
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#         return w, h
#     return None, None

# def get_random_quality():
#     return random.choice(["High Quality", "Normal", "Medium"])

# # Routes
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('error.html', message='No file uploaded')

#     file = request.files['file']
    
#     if file.filename == '' or not allowed_file(file.filename):
#         return render_template('error.html', message='Invalid file selected')

#     unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
#     upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
#     file.save(upload_path)

#     try:
#         width, height = get_dimensions(upload_path)
#         img = PILImage.create(upload_path)
#         prediction, idx, probabilities = learn.predict(img)
#         predicted_class = str(prediction).strip().lower()
        
#         fruit_quality = get_random_quality()
#         fruit_info = FRUIT_QUALITY_TABLE.get(fruit_quality.lower(), {})

#         # Save Prediction Image
#         prediction_image = f'prediction_{uuid.uuid4().hex}.png'
#         prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)

#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})\nWidth: {width}px, Height: {height}px\nQuality: {fruit_quality}")
#         plt.savefig(prediction_image_path, bbox_inches='tight')
#         plt.close()

#         # Check if the file is saved properly
#         if not os.path.exists(prediction_image_path):
#             return render_template('error.html', message="Error: Prediction image not saved!")

#         return render_template('prediction.html', 
#                                prediction_image=prediction_image, 
#                                predicted_class=predicted_class, 
#                                fruit_info=fruit_info,
#                                width=width, 
#                                height=height,
#                                fruit_quality=fruit_quality)

#     except Exception as e:
#         return render_template('error.html', message=f"Prediction error: {e}")

# @app.route('/static/<path:filename>')
# def static_file(filename):
#     return send_from_directory(STATIC_FOLDER, filename)

# # Fruit Quality Info
# FRUIT_QUALITY_TABLE = {
#     "high quality": {"Weight (g)": "200+", "Height (cm)": "8+", "Grade": "Premium (Grade A)", "Diameter (mm)": "80+"},
#     "medium": {"Weight (g)": "150-200", "Height (cm)": "6-8", "Grade": "Standard (Grade B)", "Diameter (mm)": "65-80"},
#     "normal": {"Weight (g)": "100-150", "Height (cm)": "5-6", "Grade": "Commercial (Grade C)", "Diameter (mm)": "50-65"},
# }

# if __name__ == '__main__':
#     app.run(debug=True, host='localhost', port=8000)



from flask import Flask, render_template, request, send_from_directory, jsonify
from fastai.vision.all import *
import os
import uuid
import matplotlib.pyplot as plt
import cv2
import random
from werkzeug.utils import secure_filename

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'predictions'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load Model
MODEL_PATH = r"C:\\Users\\Administrator\\Desktop\\all fruits\\My all fruit_classifier1.pk1"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)

try:
    learn = load_learner(MODEL_PATH)
    print(f"Model loaded successfully: {learn.dls.vocab}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dimensions(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return w, h
    return None, None

def get_random_quality():
    return random.choice(["High Quality", "Normal", "Medium"])

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message='No file uploaded')

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return render_template('error.html', message='Invalid file selected')

    unique_filename = secure_filename(f'{uuid.uuid4().hex}_{file.filename}')
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(upload_path)

    try:
        width, height = get_dimensions(upload_path)
        img = PILImage.create(upload_path)
        prediction, idx, probabilities = learn.predict(img)
        predicted_class = str(prediction).strip().lower()

        fruit_quality = get_random_quality()
        fruit_info = FRUIT_QUALITY_TABLE.get(fruit_quality.lower(), {})

        # Save Prediction Image
        prediction_image = f'prediction_{uuid.uuid4().hex}.png'
        prediction_image_path = os.path.join(STATIC_FOLDER, prediction_image)

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {predicted_class} (Confidence: {probabilities[idx]:.4f})\nWidth: {width}px, Height: {height}px\nQuality: {fruit_quality}")
        plt.savefig(prediction_image_path, bbox_inches='tight')
        plt.close()

        # Check if the file is saved properly
        if not os.path.exists(prediction_image_path):
            return render_template('error.html', message="Error: Prediction image not saved!")

        return render_template('prediction.html',
                               prediction_image=prediction_image,
                               predicted_class=predicted_class,
                               fruit_info=fruit_info,
                               width=width,
                               height=height,
                               fruit_quality=fruit_quality)

    except Exception as e:
        return render_template('error.html', message=f"Prediction error: {e}")
# ------------------------------------------------


# 
# @app.route('/delete_image', methods=['POST'])
def delete_image():
    # Get the image name from the request
    data = request.get_json()
    image_name = data.get('image_name')

    # Construct the full path to the image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    # Check if the image exists and delete it
    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({"message": "Image deleted successfully"}), 200
    else:
        return jsonify({"message": "Image not found"}), 404
    # -----------------------------
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.get_json()
    image_path = os.path.join(STATIC_FOLDER, data['image_name'])

    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({"status": "success", "message": "Image deleted successfully"})
    else:
        return jsonify({"status": "error", "message": "Image not found"}), 404

# Fruit Quality Info
FRUIT_QUALITY_TABLE = {
    "high quality": {"Weight (g)": "200+", "Height (cm)": "8+", "Grade": "Premium (Grade A)", "Diameter (mm)": "80+"},
    "medium": {"Weight (g)": "150-200", "Height (cm)": "6-8", "Grade": "Standard (Grade B)", "Diameter (mm)": "65-80"},
    "normal": {"Weight (g)": "100-150", "Height (cm)": "5-6", "Grade": "Commercial (Grade C)", "Diameter (mm)": "50-65"},
}

# HTML Templates
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/error.html')
def error():
    return render_template('error.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')




if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)



