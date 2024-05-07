import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS, cross_origin
import os
import werkzeug
import logging
app = Flask(__name__)
CORS(app)
# Load the rfc model
model=joblib.load("random_forest_model.pkl")
if model is None:
    print("Error: Model not found")
else:  
    print("model loaded successfully")
if not os.path.exists('./uploads'):
    os.makedirs('./uploads')
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))  # Resize to (224, 224)
    # normalize
    image = image / 255.0
    image_flat = image.flatten()  # Flatten the image
    return image_flat


class_lables={0:'Aphids',1:"Army Worm",2:"Bacterial Blight",3:"Healthy",4:"Powdery Mildew",5:"Target Spot"}

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        if request.method == 'POST' and 'image' in request.files:
            f = request.files['image']
            filename = werkzeug.utils.secure_filename(f.filename)
            file_path = os.path.join('./uploads', filename)
            f.save(file_path)
            preprocessed_image = preprocess_image(file_path)
            predicted_class = model.predict([preprocessed_image])
            response = {
                'filename': filename,
                'message': 'File uploaded successfully',
                'prediction': class_lables[predicted_class[0]]
            }
            return jsonify(response)
        else:
            return jsonify({'message': 'Invalid request. Please upload an image file.', 'status': 400})
    except Exception as e:
        return jsonify({'message': 'Error during prediction: ' + str(e), "status": 500})
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)

