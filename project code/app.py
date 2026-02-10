
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('vegetable_classifier_model.h5')

# Class mapping
class_map = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli',
    5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber',
    10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['image']
        
        # Save the file
        filename = 'temp_image.jpg'
        file.save(filename)
        
        # Load and preprocess the image
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_vegetable = class_map[predicted_class]
        confidence = float(predictions[0][predicted_class]) * 100
        
        # Clean up
        os.remove(filename)
        
        return render_template('result.html', 
                             vegetable=predicted_vegetable, 
                             confidence=f"{confidence:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
