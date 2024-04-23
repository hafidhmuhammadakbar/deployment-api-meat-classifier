from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

model = tf.keras.models.load_model('beef_pork_horse_classifier.h5')

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': 'MobileNetV3Large', 'framework': 'TensorFlow', 'task': 'Image Classification for Beef, Pork, and Horse', 'accuracy': '97.43%', 'input': 'URL', 'output': 'Predicted class and probabilities', 'model_url': 'https://www.kaggle.com/code/hafidhmuhammadakbar/mobilenetv3large-fix'})

@app.route('/models', methods=['POST'])
def predict():
    # Check if the request contains a URL
    if 'url' not in request.form:
        return jsonify({'error': 'No URL provided'})

    url = request.form['url']

    # Download the image from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({'error': f'Failed to download image from URL: {str(e)}'})

    # Resize image
    image = image.resize((224, 224))

    # Preprocess image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    probabilities = tf.reduce_max(predictions, axis=1) * 100

    class_names = ['Horse', 'Meat', 'Pork']
    predicted_class = class_names[predicted_label]
    probabilities_class = '%.2f' % probabilities.numpy()[0]

    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
