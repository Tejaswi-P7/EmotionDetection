import cv2
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN

# Load the emotion recognition model
model = load_model(r"C:\Users\Tejaswi\OneDrive\Documents\EMOTION DETECTION\Emotiondetection1\emotiondetection_model_2_accuracy_80.ipynb.h5", compile=False)

# Define the class names (replace with your actual class names)
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create an instance of the MTCNN detector
detector = MTCNN()

# Define the font and text position for displaying the predicted emotion
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (50, 50)
font_scale = 1
font_thickness = 2

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the web camera's image
    ret, frame = camera.read()

    # Check if the frame is empty
    if not ret:
        print("Error: Failed to capture frame from the webcam.")
        break

    # Detect faces in the frame using MTCNN
    results = detector.detect_faces(frame)

    # Iterate over detected faces
    for result in results:
        # Extract bounding box coordinates
        x, y, w, h = result['box']

        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Convert the face to grayscale
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize the face to match the input size of the model
        resized_face = cv2.resize(gray_face, (64, 64))  # Assuming the model expects 48x48 input size

        # Preprocess the face for the model
        resized_face = np.expand_dims(resized_face, axis=-1)  # Add channel dimension
        resized_face = np.asarray(resized_face, dtype=np.float32) / 255.0  # Normalize pixel values

        print("Resized face shape:", resized_face.shape)  # Print the shape of the resized face

        # Predict the emotion
        prediction = model.predict(np.expand_dims(resized_face, axis=0))
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display predicted emotion next to the face
        emotion_text = f"Emotion: {class_name}"
        cv2.putText(frame, emotion_text, (x, y - 10), font, font_scale, (255, 255, 255), font_thickness)

    # Show the frame
    cv2.imshow("Emotion Recognition", frame)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
