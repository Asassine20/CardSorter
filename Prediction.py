"""import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('/Users/andrewsassine/Downloads/DistortImages/saved_model')

# Load the image you want to predict
img_path = '/Users/andrewsassine/Downloads/DistortImages/PreImage/Class1/BlainesArcanine.png'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(1038, 754))

# Preprocess the image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
prediction = model.predict(img_array)

# Interpret the prediction
if prediction[0][0] > 0.5:
    print("Card")
else:
    print("Not a Card")"""
    
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('/Users/andrewsassine/Downloads/DistortImages/saved_model')

# Define the dimensions for resizing the frame
img_width = 754
img_height = 1038

# Define the labels corresponding to the predicted classes
labels = ["Absol, Pikachu, Zacian"]

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (img_width, img_height))

    # Preprocess the frame
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    prediction = model.predict(img_array)
    prediction_label = labels[int(prediction[0][0])]

    # Display the prediction label on the frame
    cv2.putText(frame, prediction_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Card Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

