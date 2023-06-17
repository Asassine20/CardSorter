import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to the train, test, and validation directories
train_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TrainPath/Class1/Images'
test_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TestPath/Class1/Images'
valid_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/ValidationPath/Class1/Images'


# Define image size and batch size
img_height = 1038
img_width = 754
batch_size = 32

# Create the ImageDataGenerator with rescaling and data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values between 0 and 1
    rotation_range=20,  # Random rotation between -20 and +20 degrees
    width_shift_range=0.2,  # Random horizontal shift by 20% of the image width
    height_shift_range=0.2,  # Random vertical shift by 20% of the image height
    shear_range=0.2,  # Random shearing transformations
    zoom_range=0.2,  # Random zoom in/out by 20%
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill any newly created pixels after rotation or shifting
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for test data

# Load and preprocess the train, test, and validation data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the number of classes
num_classes = train_data.num_classes

# Print the class labels
class_labels = list(train_data.class_indices.keys())
print("Class Labels:", class_labels)
