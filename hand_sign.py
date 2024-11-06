import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import string  

# Load the trained model
model = tf.keras.models.load_model('D:/tensorrrr/hand_sign_model.h5')  

# Define the labels (corresponding to the alphabet)
labels = list(string.ascii_lowercase)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture image")
        continue  

    roi = frame[100:400, 100:400] 

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0 
    reshaped = np.reshape(normalized, (1, 28, 28, 1)) 


    prediction = model.predict(reshaped)
    predicted_label = labels[np.argmax(prediction)]


    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)  
    cv2.imshow('Hand Sign Prediction', frame)


    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()