from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
from mtcnn import MTCNN
from flask_cors import CORS

app = Flask("EmotionDetectionAPI")
CORS(app)

# Load the emotion recognition model
model = load_model(r"C:\Users\Tejaswi\OneDrive\Documents\EMOTION DETECTION\Emotiondetection1\emotiondetection_model_2_accuracy_80.ipynb.h5", compile=False)

# Define the class names
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create an instance of the MTCNN detector
detector = MTCNN()

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read the image file
    image_file = request.files['image']
    
    # Convert the image file to a numpy array
    image_np = np.frombuffer(image_file.read(), np.uint8)
    
    # Decode the image
    try:
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'Error decoding image', 'details': str(e)}), 400
    
    # Detect faces in the image
    results = detector.detect_faces(image)
    
    # Perform emotion detection for each detected face
    detected_emotions = []
    for result in results:
        x, y, w, h = result['box']
        face = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (64, 64))
        resized_face = np.expand_dims(resized_face, axis=-1)
        resized_face = np.asarray(resized_face, dtype=np.float32) / 255.0
        prediction = model.predict(np.expand_dims(resized_face, axis=0))
        index = np.argmax(prediction)
        class_name = class_names[index]
        detected_emotions.append({
            'emotion': class_name,
            'box': [x, y, w, h]
        })
    
    return jsonify(detected_emotions)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
