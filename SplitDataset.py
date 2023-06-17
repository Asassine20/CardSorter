import os
import random
import shutil

def split_dataset(images_dir, train_dir, test_dir, validation_dir, split_ratio):
    # Create the output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Get the list of set directories
    set_dirs = os.listdir(images_dir)
    
    for set_dir in set_dirs:
        # Path to the current set's images
        set_images_dir = os.path.join(images_dir, set_dir)
        
        # Skip non-directory files
        if not os.path.isdir(set_images_dir):
            continue
        
        # Path to the current set's train, test, and validation directories
        set_train_dir = os.path.join(train_dir, set_dir)
        set_test_dir = os.path.join(test_dir, set_dir)
        set_validation_dir = os.path.join(validation_dir, set_dir)
        
        # Create the set's train, test, and validation directories if they don't exist
        os.makedirs(set_train_dir, exist_ok=True)
        os.makedirs(set_test_dir, exist_ok=True)
        os.makedirs(set_validation_dir, exist_ok=True)

        # Get the list of image files for the current set
        image_files = os.listdir(set_images_dir)
        
        # Shuffle the image files
        random.shuffle(image_files)
        
        # Calculate the number of images for each split
        num_images = len(image_files)
        num_train = int(num_images * split_ratio[0])
        num_test = int(num_images * split_ratio[1])
        num_validation = num_images - num_train - num_test
        
        # Split the images into train, test, and validation sets
        train_images = image_files[:num_train]
        test_images = image_files[num_train:num_train+num_test]
        validation_images = image_files[num_train+num_test:]
        
        # Move the images to the corresponding directories
        move_images(train_images, set_images_dir, set_train_dir)
        move_images(test_images, set_images_dir, set_test_dir)
        move_images(validation_images, set_images_dir, set_validation_dir)
        
    print("Dataset split completed successfully.")

def move_images(image_list, source_dir, target_dir):
    for image in image_list:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.move(source_path, target_path)

# Set the paths for the images and the output directories
images_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/PostImage'
train_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TrainPath'
test_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/TestPath'
validation_dir = '/Users/andrewsassine/Downloads/DistortImages/DataBase/ValidationPath'

# Define the split ratio for train, test, and validation sets (e.g., 70%, 15%, 15%)
split_ratio = (0.7, 0.15, 0.15)

# Split the dataset
split_dataset(images_dir, train_dir, test_dir, validation_dir, split_ratio)

print("Train directory:", train_dir)
print("Test directory:", test_dir)
print("Validation directory:", validation_dir)



