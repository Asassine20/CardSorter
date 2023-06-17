import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set the paths to the train, test, and validation directories
train_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TrainPath'
test_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TestPath'
valid_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/ValidationPath'

# Define image size and batch size
img_height = 1038
img_width = 754
batch_size = 32

# Create the ImageDataGenerator with rescaling
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the train, test, and validation data
train_data = datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_data = datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
valid_data = datagen.flow_from_directory(valid_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

# Define the model architecture
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# Fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
num_classes = 3 # Replace with the actual number of classes
model.add(Dense(num_classes, activation='softmax'))  # Update the number of units

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
num_epochs = 10
model.fit(train_data, epochs=num_epochs, validation_data=valid_data)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions on test data
predictions = model.predict(test_data)

model.save('/Users/andrewsassine/Downloads/DistortImages/saved_model')
