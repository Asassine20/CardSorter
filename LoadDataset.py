import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to the train, test, and validation directories
train_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TrainPath'
test_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TestPath'
valid_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/ValidationPath'

image_size = (1038, 754)
batch_size = 32

# Create the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255)

img_height = 1038  # Specify the desired image height
img_width = 754  # Specify the desired image width


# Load the train images
train_data = ImageDataGenerator().flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_data = ImageDataGenerator().flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_data = ImageDataGenerator().flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# Get the number of classes
num_classes = len(train_data.class_indices)

# Print the number of samples

print("Train samples:", train_data.samples)
print("Test samples:", test_data.samples)
print("Validation samples:", val_data.samples)
print("Number of classes:", num_classes)

