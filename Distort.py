import cv2
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def distort_image(image_path, output_dir, num_copies):
    # Extract the card name from the image file path
    card_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load the image
    image = cv2.imread(image_path)

    # Apply the distortion and create multiple copies
    for i in range(num_copies):
        # Create a copy of the original image
        distorted_image = np.copy(image)
        
        # Apply random distortions to the image
        distorted_image = apply_random_distortions(distorted_image)
        
        # Save the distorted image with the card name as the filename
        output_filename = f"{card_name}_distorted_{i}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, distorted_image)
        print(f"Distorted image saved: {output_path}")

def apply_random_distortions(image):
    # Define the distortion parameters
    scale_range = (0.8, 1.2)  # Range for scaling factors
    rotation_angle_range = (-15, 15)  # Range for rotation angles in degrees
    translation_shift_range = (-10, 10)  # Range for translation shifts in pixels
    blur_range = (3, 7)  # Range for blur kernel size
    noise_range = (10, 30)  # Range for noise strength
    brightness_range = (-50, 50)  # Range for brightness adjustment

    # Randomly select distortion parameters for this copy
    scale_factor = random.uniform(*scale_range)
    rotation_angle = random.uniform(*rotation_angle_range)
    translation_shift_x = random.uniform(*translation_shift_range)
    translation_shift_y = random.uniform(*translation_shift_range)
    blur_kernel_size = random.randint(*blur_range) * 2 + 1  # Odd kernel size
    noise_strength = random.randint(*noise_range)  # Integer noise strength
    brightness_adjustment = random.randint(*brightness_range)  # Integer brightness adjustment

    # Apply the distortions to the image
    distorted_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    rows, cols, _ = distorted_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    distorted_image = cv2.warpAffine(distorted_image, M, (cols, rows))
    M = np.float32([[1, 0, translation_shift_x], [0, 1, translation_shift_y]])
    distorted_image = cv2.warpAffine(distorted_image, M, (cols, rows))
    distorted_image = cv2.GaussianBlur(distorted_image, (blur_kernel_size, blur_kernel_size), 0)
    noise = np.random.normal(0, noise_strength, distorted_image.shape).astype(np.uint8)
    distorted_image = cv2.add(distorted_image, noise)
    distorted_image = cv2.add(distorted_image, brightness_adjustment)

    return distorted_image



# Path to the input images folder
input_folder = "/Users/andrewsassine/Downloads/DistortImages/PreImage/Class1/CrownZenith"

# Path to the output directory for distorted images
output_dir = "/Users/andrewsassine/Downloads/DistortImages/DataBase/PostImage/CrownZenith"

# Number of copies to create for each image
num_copies = 1

# Iterate through all the image files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_path = os.path.join(input_folder, filename)
        # Generate distorted images for all sets
        distort_image(image_path, output_dir, num_copies)

# Define image size and batch size for training
img_height = 1038
img_width = 754
batch_size = 32

# Create the ImageDataGenerator with rescaling and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values between 0 and 1
    rotation_range=20,  # Random rotation between -20 and +20 degrees
    width_shift_range=0.2,  # Random horizontal shift by up to 20% of the image width
    height_shift_range=0.2,  # Random vertical shift by up to 20% of the image height
    zoom_range=0.2,  # Random zoom between 80% and 120% of original size
    horizontal_flip=True,  # Random horizontal flip
    vertical_flip=True  # Random vertical flip
)

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    output_dir,  # Directory containing the distorted images
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the number of classes from the train_generator
num_classes = train_generator.num_classes

"""def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Build and compile your model using the appropriate architecture and loss function
model = build_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)"""

