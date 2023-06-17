import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/Users/andrewsassine/Downloads/DistortImages/saved_model/saved_model.pb')

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  # Normalize pixel values between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# List of class labels
class_labels = ['Class1', 'Class2', 'Class3', ...]  # Replace with your class labels

# Function to predict the class of an image
def predict_image_class(image_path):
    # Preprocess the input image
    image = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    
    # Get the confidence score for the predicted class
    confidence = prediction[0][predicted_class_index]
    
    return predicted_class_label, confidence

# Test the model on new images
test_image_paths = ['/Users/andrewsassine/Downloads/DistortImages/PreImage/Class1/CrownZenith/en_US-CZ-GG061-gardenias_vigor (1).jpg',
                    '/Users/andrewsassine/Downloads/DistortImages/PreImage/Class1/CrownZenith/en_US-CZ-082-dragalge (1).jpg',
                    '/Users/andrewsassine/Downloads/DistortImages/PreImage/Class1/CrownZenith/en_US-CZ-055-zeraora_vstar (1).jpg']  # Replace with your test image paths

for image_path in test_image_paths:
    predicted_class, confidence = predict_image_class(image_path)
    print(f'Image: {image_path}')
    print(f'Predicted Class: {predicted_class}')
    print(f'Confidence: {confidence}')
    print('---')
