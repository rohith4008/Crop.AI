from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import joblib
import os

from utils import model as model_utils
from utils import fertilizer as fert_utils
from utils import disease as disease_utils

app = Flask(__name__)

# Load crop recommendation model
crop_model = joblib.load('models/RandomForest.pkl')

# Load fertilizer dataset
fertilizer_data = pd.read_csv('Data/fertilizer.csv')

# Load plant disease detection model
device = torch.device("cpu")
disease_model = model_utils.ResNet9(3, 38)
disease_model.load_state_dict(torch.load("models/plant_disease_model.pth", map_location=device))
disease_model.eval()

disease_classes = disease_utils.disease_classes

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorus'])
            K = int(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(data)
            crop_name = prediction[0].capitalize()

            return render_template('crop-result.html', prediction=crop_name)
        except Exception as e:
            return f"Error: {e}"

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        crop_name = request.form['cropname']
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])

        recommendation = fert_utils.fertilizer_recommendation(crop_name, N, P, K, fertilizer_data)
        return render_template('fertilizer-result.html', recommendation=recommendation)
    except Exception as e:
        return f"Error: {e}"

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return "No image uploaded"

    file = request.files['image']
    if file.filename == '':
        return "Empty file name"

    try:
        img = Image.open(file)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = disease_model(img_tensor)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()
            class_name = disease_classes[class_idx]

        return render_template('disease-result.html', prediction=class_name)
    except Exception as e:
        return f"Error: {e}"
