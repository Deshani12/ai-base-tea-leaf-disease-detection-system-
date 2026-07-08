from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and class labels
model = tf.keras.models.load_model('tea_leaf_disease_model.h5')
with open('class_labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f]

# Disease information dictionary
disease_info = {
    'Tea_algal_leaf_spot': {
        'description': 'Caused by Cephaleuros virescens, characterized by green or brown velvety spots on leaves, often on older leaves.',
        'recommendation': 'Apply copper-based fungicides (e.g., Bordeaux mixture) at 0.5% concentration every 14 days during wet seasons. Improve air circulation by pruning dense foliage. Remove and destroy affected leaves to reduce spore spread.',
        'details': {
            'treatment': 'Spray copper oxychloride (0.5%) or Bordeaux mixture (1:1:100) every 2 weeks.',
            'prevention': 'Ensure proper spacing between plants, avoid overhead irrigation, and apply neem oil as a preventive spray monthly.'
        }
    },
    'Brown_Blight': {
        'description': 'Fungal disease caused by Colletotrichum spp., causing brown to black lesions with gray centers on leaves.',
        'recommendation': 'Use fungicides like mancozeb (0.2%) or carbendazim (0.1%) every 10–14 days. Remove and burn infected leaves to prevent spread. Maintain field hygiene by clearing debris.',
        'details': {
            'treatment': 'Apply mancozeb at 2 g/L or carbendazim at 1 g/L, starting at disease onset.',
            'prevention': 'Rotate fungicides to prevent resistance, ensure well-drained soil, and avoid waterlogging.'
        }
    },
    'Gray_Blight': {
        'description': 'Caused by Pestalotiopsis spp., leads to grayish-white patches on leaves, often with black fruiting bodies.',
        'recommendation': 'Apply systemic fungicides like thiophanate-methyl (0.1%) every 15 days. Improve plant vigor with balanced fertilization. Avoid wounding leaves during plucking.',
        'details': {
            'treatment': 'Spray thiophanate-methyl at 1 g/L or azoxystrobin at 0.5 g/L.',
            'prevention': 'Use disease-free planting material, maintain optimal nitrogen levels, and prune regularly.'
        }
    },
    'Helopeltis': {
        'description': 'Damage by Helopeltis theivora (tea mosquito bug), causing dark spots, leaf distortion, and reduced photosynthesis.',
        'recommendation': 'Use insecticides like imidacloprid (0.03%) or thiacloprid (0.04%) during early morning. Monitor pest populations with sticky traps. Encourage natural predators like spiders.',
        'details': {
            'treatment': 'Apply imidacloprid at 0.3 mL/L or neem-based formulations (5 mL/L).',
            'prevention': 'Remove weeds, use shade trees to reduce pest habitat, and apply insect-repellent plants like marigold.'
        }
    },
    'Red_Spider': {
        'description': 'Infestation by red spider mites (Oligonychus coffeae), causing stippling, yellowing, and bronze-colored leaves.',
        'recommendation': 'Apply miticides like dicofol (0.05%) or abamectin (0.01%) every 10 days. Maintain humidity above 60% to deter mites. Avoid excessive nitrogen fertilizers.',
        'details': {
            'treatment': 'Spray dicofol at 0.5 mL/L or sulphur-based miticides (2 g/L).',
            'prevention': 'Introduce predatory mites, maintain soil moisture, and avoid dusty conditions.'
        }
    },
    'Green_Mirid_Bug': {
        'description': 'Damage by green mirid bugs, leading to leaf curling, brown spots, and reduced growth.',
        'recommendation': 'Use neem-based insecticides (5 mL/L) or quinalphos (0.05%) during pest activity. Introduce natural enemies like ladybugs. Monitor with pheromone traps.',
        'details': {
            'treatment': 'Apply quinalphos at 0.5 mL/L or neem oil (5 mL/L) in the evening.',
            'prevention': 'Plant trap crops like marigold, maintain biodiversity, and avoid broad-spectrum pesticides.'
        }
    },
    'Healthy_Leaf': {
        'description': 'No disease detected; leaves appear green, glossy, and undamaged.',
        'recommendation': 'Continue regular monitoring for early signs of pests or diseases. Apply balanced fertilizers (e.g., NPK 20:10:10) monthly. Ensure proper irrigation and pruning.',
        'details': {
            'treatment': 'No treatment required.',
            'prevention': 'Conduct weekly inspections, use organic mulch to retain soil moisture, and apply preventive neem sprays (3 mL/L) monthly.'
        }
    }
}

# Initialize SQLite database for prediction history
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, prediction TEXT, probability REAL, timestamp TEXT, image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = image.convert('RGB')  # Ensure RGB format for PNGs with alpha
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    # Server-side validation
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', error='Only PNG and JPEG images are supported')
    file_data = file.read()
    if len(file_data) > 5 * 1024 * 1024:  # 5MB limit
        return render_template('index.html', error='Image size must be less than 5MB')
    
    try:
        # Save image to uploads folder
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(image_path, 'wb') as f:
            f.write(file_data)
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(file_data))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        probability = float(predictions[0][predicted_class]) * 100
        prediction = class_labels[predicted_class]
        
        # Get all probabilities for visualization
        all_probabilities = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
        probability_labels = list(all_probabilities.keys())
        probability_values = list(all_probabilities.values())
        
        # Save to prediction history
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("INSERT INTO predictions (prediction, probability, timestamp, image_path) VALUES (?, ?, ?, ?)",
                  (prediction, probability, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_path))
        conn.commit()
        conn.close()
        
        # Get disease info
        description = disease_info.get(prediction, {}).get('description', 'No information available')
        recommendation = disease_info.get(prediction, {}).get('recommendation', 'No recommendations available')
        details = disease_info.get(prediction, {}).get('details', {})
        
        return render_template('index.html', 
                             prediction=prediction, 
                             probability=probability, 
                             probability_labels=probability_labels,
                             probability_values=probability_values,
                             description=description,
                             recommendation=recommendation,
                             details=details,
                             image_path=image_path)
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

@app.route('/history')
def history():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT prediction, probability, timestamp, image_path FROM predictions ORDER BY timestamp DESC")
    history = c.fetchall()
    conn.close()
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)